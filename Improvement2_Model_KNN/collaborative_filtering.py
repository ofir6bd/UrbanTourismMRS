import copy
import cornac
import itertools
import numpy as np
from utils import argnonz, argtopk, rand_member_nb, rand_member_arr_nb, myrecall
from typing import Union
from sklearn.metrics.pairwise import cosine_similarity
from cornac.models import Recommender
from cornac.exception import ScoreException


SEED = 2024
TOPK = 10
HIDDEN_SIZE = 32


def formatdict(d):
    s = " ".join(f"{k}={v:>7.4f}" if not isinstance(v, str) else f"{k}={v}" for k, v in d.items())
    s = "{" + s + "}"
    return s


# -----------------------------------------------------------------------------


def isOK(y, uid_map, iid_map):
    assert set(np.unique(y)) == {0,1}
    assert y.shape[0] == len(uid_map)
    assert y.shape[1] == len(iid_map)
    assert list(uid_map.keys()) == list(uid_map.values())
    assert list(iid_map.keys()) == list(iid_map.values())
    return True


def prepare_positive_feedback_data(y, uid_map, iid_map):  # --> only positive feedback
    assert isOK(y, uid_map, iid_map)
    N, J = y.shape
    n_implicit_feedback = y.sum()
    uir_tup = (
        np.zeros(n_implicit_feedback, dtype="i8"),  # user_indices
        np.zeros(n_implicit_feedback, dtype="i8"),  # item_indices
        np.zeros(n_implicit_feedback, dtype="f8"),  # feedback values (ratings)
    )
    counter = 0
    for n in range(N):
        for j in range(J):
            if y[n,j] != 0:
                uir_tup[0][counter] = uid_map[n]
                uir_tup[1][counter] = iid_map[j]
                uir_tup[2][counter] = 1
                counter += 1
    assert n_implicit_feedback == counter, "Implicit feedback mismatch detected"
    data = cornac.data.Dataset(num_users=N, num_items=J, uid_map=uid_map, iid_map=iid_map, uir_tuple=uir_tup, seed=SEED)
    return data


def prepare_and_split_positive_feedback_data(y, uid_map, iid_map, te_frac=0.2):  # --> w\i user train-test items split
    assert isOK(y, uid_map, iid_map)
    assert (0 < te_frac < 1)
    N = len(y)
    tr_y = np.zeros(y.shape, dtype="i8")
    te_y = np.zeros(y.shape, dtype="i8")
    tr_frac = 1 - te_frac
    for n in range(N):
        pos = argnonz(y[n,:])
        tr_pos = rand_member_arr_nb(pos, size=max(1, int(len(pos) * tr_frac)))
        te_pos = np.setdiff1d(pos, tr_pos)
        tr_y[n,tr_pos] = 1
        te_y[n,te_pos] = 1
    return (
        prepare_positive_feedback_data(tr_y, uid_map, iid_map),
        prepare_positive_feedback_data(te_y, uid_map, iid_map),
    )


# -----------------------------------------------------------------------------


def hsearch(
        model: cornac.models.Recommender, space: dict[str, Union[np.ndarray, list]], space_tied_constraints: dict[str,str],
        tr_data: cornac.data.Dataset,
        te_data: cornac.data.Dataset,
        gridsearch: bool = True, n_iters: int = 0,
):

    assert (tr_data.num_users == te_data.num_users) and (tr_data.uid_map == te_data.uid_map), "Shape or Map mismatch"
    assert (tr_data.num_items == te_data.num_items) and (tr_data.iid_map == te_data.iid_map), "Shape or Map mismatch"

    # grid search hyperparameters optimisation or not?
    __inp = copy.deepcopy(space)
    if gridsearch:
        space = (dict(zip(__inp.keys(), values)) for values in itertools.product(*__inp.values()))
    else:
        space = (__inp)

    # init optimisation state
    best_params = dict()
    best_recsys = None
    best_recall = 0
    tr_data_csr = tr_data.csr_matrix  # for fast row access
    te_data_csr = te_data.csr_matrix  # for fast row access

    acc = 0
    while 1:
        acc += 1

        # sample hyperparameters dict
        if gridsearch:
            try:
                params = next(space)
            except StopIteration:
                break
        else:
            if acc >= n_iters:
                break
            params = {k: rand_member_nb(space[k]) for k in space.keys()}

        # update tied hyperparameters
        for k, k_tied in space_tied_constraints.items():
            params[k] = params[k_tied]

        # optimise
        recsys = model.clone(params)
        recsys.fit(tr_data, te_data)
        # estimate performance for each user in test data
        Q = []
        scores = np.zeros(tr_data.num_items, dtype="f8")
        for n in range(te_data.num_users):
            tr_cn_true = tr_data_csr.getrow(n).indices.astype("i8")
            te_cn_true = te_data_csr.getrow(n).indices.astype("i8")
            # get scores and down weight already chosen train items
            scores[:] = np.squeeze(recsys.score(n))
            scores[tr_cn_true] = -100000
            # calculate test items
            te_cn_pred = argtopk(scores, k=max(TOPK, len(te_cn_true)))
            Q.append(myrecall(te_cn_true, te_cn_pred))

        if (recall := np.mean(Q)) > best_recall:
            best_params = params
            best_recsys = recsys
            best_recall = recall
        del recsys, Q
        print(f"recall={recall:.4f}  "
              f"params={formatdict(params)}  |  best_recall={best_recall:.4f}  best_params={formatdict(best_params)}")

    return best_params, best_recsys, best_recall


# -----------------------------------------------------------------------------


_pop = cornac.models.MostPop(
    name="POP"
)
_pop_search_space = {"name": ["POP"]}
_pop_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_wmf = cornac.models.WMF(
    name="WMF",
    k=HIDDEN_SIZE,
    lambda_u=0.01,
    lambda_v=0.01,
    a=1,                     # the confidence (c_nj) of collected ratings a.k.a. positive feedback
    b=0.01,                  # the confidence (c_nj) of unseen ratings
    learning_rate=1e-3,
    batch_size=96,           # how many random unique items to consider in one batch (num_users x batch_size)
    max_iter=1000,
    verbose=0,
    seed=SEED,
)

_wmf_search_space = {
    "lambda_u"     : np.logspace(-2.0, 2.0, num=10, base=10),
    "b"            : np.logspace(-4.0, 0.0, num=10, base=10),
}
_wmf_search_space_tied_constraints = {"lambda_v": "lambda_u"}  # tied parameter


# -----------------------------------------------------------------------------


_bpr = cornac.models.BPR(
    name="BPR",
    k=HIDDEN_SIZE,
    use_bias=False,
    lambda_reg=0.01,
    learning_rate=1e-3,
    max_iter=8000,           # should be linearly proportional to a nnz in data
    verbose=0,
    num_threads=1,
    seed=SEED,
)

_bpr_search_space = {"lambda_reg": np.logspace(-4.0, 0.0, num=20, base=10)}
_bpr_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_vae = cornac.models.VAECF(
    name="VAE",
    k=HIDDEN_SIZE,
    autoencoder_structure=[HIDDEN_SIZE],
    act_fn="tanh",
    likelihood="mult",
    n_epochs=1000,
    batch_size=256,          # how many random unique users to consider in one batch (batch_size x num_items)
    learning_rate=1e-3,
    beta=1.0,
    verbose=0,
    seed=SEED,
)

_vae_search_space = {"beta": np.linspace(0.5, 1.5, num=20)}
_vae_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_ease = cornac.models.EASE(
    name="EASE",
    lamb=1.0,
    posB=False,
    verbose=0,
    seed=SEED,
)

_ease_search_space = {"lamb": np.logspace(2.0, 2.9, num=20, base=10)}
_ease_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------


_ngcf = cornac.models.NGCF(
    name="NGCF",
    emb_size=HIDDEN_SIZE,
    layer_sizes=[HIDDEN_SIZE],
    dropout_rates=[0.1],
    num_epochs=250,
    learning_rate=1e-3,
    batch_size=512,
    early_stopping=None,
    lambda_reg=0.01,
    verbose=0,
    seed=SEED,
)

_ngcf_search_space = {"lambda_reg": np.logspace(-4.0, 0.0, num=20, base=10)}
_ngcf_search_space_tied_constraints = {}


# -----------------------------------------------------------------------------
class KNNBaseline(Recommender):
    """KNNBaseline implementation based on the paper's recommendations"""
    
    def __init__(
        self,
        name="KNNBaseline",
        k=21,  # Optimal K from paper
        trainable=True,
        verbose=False,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.seed = seed
        
        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Will be set during fit
        self.user_item_matrix = None
        self.user_similarities = None
        self.user_baselines = None
        self.item_baselines = None
        self.global_mean = None
        
    def fit(self, train_set, val_set=None):
        """Fit the KNNBaseline model"""
        super().fit(train_set, val_set)
        
        # Set random seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Keep as sparse matrix for efficiency
        self.user_item_matrix = train_set.csr_matrix
        
        # CHANGE 1: Fix global mean calculation (include zeros)
        total_interactions = self.user_item_matrix.sum()
        total_possible = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        self.global_mean = total_interactions / total_possible
        
        # CHANGE 2: Fix user baselines (include zeros in denominator)
        user_sums = np.array(self.user_item_matrix.sum(axis=1)).flatten()
        user_means = user_sums / self.user_item_matrix.shape[1]
        self.user_baselines = user_means - self.global_mean
        
        # CHANGE 3: Fix item baselines (include zeros in denominator)
        item_sums = np.array(self.user_item_matrix.sum(axis=0)).flatten()
        item_means = item_sums / self.user_item_matrix.shape[0]
        self.item_baselines = item_means - self.global_mean
        
        # Pre-compute user similarities using cosine similarity (as recommended in paper)
        self.user_similarities = cosine_similarity(self.user_item_matrix)
        np.fill_diagonal(self.user_similarities, 0)  # Remove self-similarity
        
        return self
    
    def score(self, user_idx, item_idx=None):
        """Score items for a user"""
        if item_idx is not None:
            return self._score_single_item(user_idx, item_idx)
        else:
            return self._score_all_items(user_idx)
    
    def _score_single_item(self, user_idx, item_idx):
        """Score a single item for a user"""
        all_scores = self._score_all_items(user_idx)
        return all_scores[item_idx]
    
    def _score_all_items(self, user_idx):
        """Score all items for a user using KNNBaseline approach"""
        if user_idx >= self.user_item_matrix.shape[0]:
            raise ScoreException(f"User index {user_idx} is out of range")
        
        # Get user similarities and find top-k neighbors
        user_similarities = self.user_similarities[user_idx]
        
        # Get top-k similar users (excluding self)
        top_k_indices = np.argsort(user_similarities)[-self.k:]
        top_k_similarities = user_similarities[top_k_indices]
        
        # Filter out users with very low similarity
        valid_mask = top_k_similarities > 1e-8
        if np.any(valid_mask):
            valid_indices = top_k_indices[valid_mask]
            valid_similarities = top_k_similarities[valid_mask]
        else:
            # Fallback: use baseline prediction only
            baseline_scores = self.global_mean + self.user_baselines[user_idx] + self.item_baselines
            # CHANGE 4: Removed sustainability boost - return pure baseline
            return np.maximum(baseline_scores, 0)
        
        # Calculate baseline predictions for all items
        baseline_scores = self.global_mean + self.user_baselines[user_idx] + self.item_baselines
        
        # Calculate KNN-based adjustments
        num_items = self.user_item_matrix.shape[1]
        knn_adjustments = np.zeros(num_items)
        
        if len(valid_similarities) > 0:
            # Normalize similarities to use as weights
            similarity_sum = np.sum(np.abs(valid_similarities))
            if similarity_sum > 1e-8:
                similarity_weights = valid_similarities / similarity_sum
                
                # For each item, calculate weighted deviation from baseline
                for item_idx in range(num_items):
                    weighted_deviation = 0.0
                    weight_sum = 0.0
                    
                    for neighbor_idx, weight in zip(valid_indices, similarity_weights):
                        # Check if neighbor rated this item
                        if self.user_item_matrix[neighbor_idx, item_idx] > 0:
                            neighbor_rating = self.user_item_matrix[neighbor_idx, item_idx]
                            neighbor_baseline = (self.global_mean + 
                                               self.user_baselines[neighbor_idx] + 
                                               self.item_baselines[item_idx])
                            deviation = neighbor_rating - neighbor_baseline
                            weighted_deviation += weight * deviation
                            weight_sum += abs(weight)
                    
                    if weight_sum > 1e-8:
                        knn_adjustments[item_idx] = weighted_deviation / weight_sum
        
        # Combine baseline + KNN adjustment
        scores = baseline_scores + knn_adjustments
        
        # CHANGE 4: Removed sustainability boost completely
        # Pure KNN-Baseline scores without domain-specific modifications
        
        # Ensure scores are non-negative
        scores = np.maximum(scores, 0)
        
        return scores

# Create the model instance with paper's optimal parameters
_knn_baseline = KNNBaseline(
    name="KNNBaseline",
    k=21,  # Optimal K from paper
    verbose=0,
    seed=SEED,
)

# Single set of parameters based on paper's findings
_knn_baseline_search_space = {
    "name": ["KNNBaseline"]  # No hyperparameter search needed
}

_knn_baseline_search_space_tied_constraints = {}



# -----------------------------------------------------------------------------

# Keep your existing hybrid models for comparison
class HybridKNN:
    def __init__(self, name="HybridKNN", k_users=50, k_items=50, alpha=0.5, seed=SEED):
        self.name = name
        self.k_users = k_users
        self.k_items = k_items
        self.alpha = alpha
        self.seed = seed
        self.user_sim = None
        self.item_sim = None
        self.train_matrix = None
        # Add missing attributes to match Cornac interface
        self.num_users = None
        self.num_items = None
        self.uid_map = None
        self.iid_map = None
        
    def clone(self, params):
        new_model = HybridKNN(**params)
        return new_model
        
    def fit(self, train_data, val_data=None):
        # Convert to CSR matrix for efficient operations
        self.train_matrix = train_data.csr_matrix
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        # Copy the mapping attributes from train_data
        self.uid_map = train_data.uid_map
        self.iid_map = train_data.iid_map
        
        # Compute user-user similarity
        self.user_sim = cosine_similarity(self.train_matrix)
        np.fill_diagonal(self.user_sim, 0)  # Remove self-similarity
        
        # Compute item-item similarity  
        self.item_sim = cosine_similarity(self.train_matrix.T)
        np.fill_diagonal(self.item_sim, 0)  # Remove self-similarity
        
    def score(self, user_id):
        scores = np.zeros(self.num_items)
        
        # User-based prediction
        user_similarities = self.user_sim[user_id]
        top_users = np.argsort(user_similarities)[-self.k_users:]
        
        if len(top_users) > 0 and np.sum(np.abs(user_similarities[top_users])) > 1e-8:
            sim_weights = user_similarities[top_users]
            sim_weights = sim_weights / (np.sum(np.abs(sim_weights)) + 1e-8)
            user_scores = np.average(self.train_matrix[top_users].toarray(), 
                                   weights=np.abs(sim_weights), axis=0)
        else:
            user_scores = np.zeros(self.num_items)
            
        # Item-based prediction
        user_items = self.train_matrix[user_id].indices
        item_scores = np.zeros(self.num_items)
        
        if len(user_items) > 0:
            for item_id in range(self.num_items):
                item_similarities = self.item_sim[item_id, user_items]
                if len(item_similarities) > 0:
                    top_items_mask = np.argsort(item_similarities)[-min(self.k_items, len(item_similarities)):]
                    
                    if len(top_items_mask) > 0:
                        weights = item_similarities[top_items_mask]
                        if np.sum(np.abs(weights)) > 1e-8:  # Check if weights are non-zero
                            weights = weights / (np.sum(np.abs(weights)) + 1e-8)
                            item_scores[item_id] = np.average(
                                self.train_matrix[user_id, user_items[top_items_mask]].toarray().flatten(),
                                weights=np.abs(weights)
                            )
        
        # Hybrid combination
        scores = self.alpha * user_scores + (1 - self.alpha) * item_scores
        return scores

_hybrid_knn = HybridKNN(
    name="HybridKNN",
    k_users=30,
    k_items=30, 
    alpha=0.5,
    seed=SEED,
)

_hybrid_knn_search_space = {
    "k_users": [30],
    "k_items": [30],
    "alpha": [0.5]
}
_hybrid_knn_search_space_tied_constraints = {}

# -----------------------------------------------------------------------------

# Keep your existing hybrid models for comparison
class HybridKNN25:
    def __init__(self, name="HybridKNN25", k_users=50, k_items=50, alpha=0.5, seed=SEED):
        self.name = name
        self.k_users = k_users
        self.k_items = k_items
        self.alpha = alpha
        self.seed = seed
        self.user_sim = None
        self.item_sim = None
        self.train_matrix = None
        # Add missing attributes to match Cornac interface
        self.num_users = None
        self.num_items = None
        self.uid_map = None
        self.iid_map = None
        
    def clone(self, params):
        new_model = HybridKNN25(**params)
        return new_model
        
    def fit(self, train_data, val_data=None):
        # Convert to CSR matrix for efficient operations
        self.train_matrix = train_data.csr_matrix
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        # Copy the mapping attributes from train_data
        self.uid_map = train_data.uid_map
        self.iid_map = train_data.iid_map
        
        # Compute user-user similarity
        self.user_sim = cosine_similarity(self.train_matrix)
        np.fill_diagonal(self.user_sim, 0)  # Remove self-similarity
        
        # Compute item-item similarity  
        self.item_sim = cosine_similarity(self.train_matrix.T)
        np.fill_diagonal(self.item_sim, 0)  # Remove self-similarity
        
    def score(self, user_id):
        scores = np.zeros(self.num_items)
        
        # User-based prediction
        user_similarities = self.user_sim[user_id]
        top_users = np.argsort(user_similarities)[-self.k_users:]
        
        if len(top_users) > 0 and np.sum(np.abs(user_similarities[top_users])) > 1e-8:
            sim_weights = user_similarities[top_users]
            sim_weights = sim_weights / (np.sum(np.abs(sim_weights)) + 1e-8)
            user_scores = np.average(self.train_matrix[top_users].toarray(), 
                                   weights=np.abs(sim_weights), axis=0)
        else:
            user_scores = np.zeros(self.num_items)
            
        # Item-based prediction
        user_items = self.train_matrix[user_id].indices
        item_scores = np.zeros(self.num_items)
        
        if len(user_items) > 0:
            for item_id in range(self.num_items):
                item_similarities = self.item_sim[item_id, user_items]
                if len(item_similarities) > 0:
                    top_items_mask = np.argsort(item_similarities)[-min(self.k_items, len(item_similarities)):]
                    
                    if len(top_items_mask) > 0:
                        weights = item_similarities[top_items_mask]
                        if np.sum(np.abs(weights)) > 1e-8:  # Check if weights are non-zero
                            weights = weights / (np.sum(np.abs(weights)) + 1e-8)
                            item_scores[item_id] = np.average(
                                self.train_matrix[user_id, user_items[top_items_mask]].toarray().flatten(),
                                weights=np.abs(weights)
                            )
        
        # Hybrid combination
        scores = self.alpha * user_scores + (1 - self.alpha) * item_scores
        return scores

_hybrid_knn_25 = HybridKNN25(
    name="HybridKNN25",
    k_users=30,
    k_items=30, 
    alpha=0.25,
    seed=SEED,
)

_hybrid_knn_25_search_space = {
    "k_users": [30],
    "k_items": [30],
    "alpha": [0.25]
}
_hybrid_knn_25_search_space_tied_constraints = {}



# -----------------------------------------------------------------------------

# Keep your existing hybrid models for comparison
class HybridKNN75:
    def __init__(self, name="HybridKNN75", k_users=50, k_items=50, alpha=0.5, seed=SEED):
        self.name = name
        self.k_users = k_users
        self.k_items = k_items
        self.alpha = alpha
        self.seed = seed
        self.user_sim = None
        self.item_sim = None
        self.train_matrix = None
        # Add missing attributes to match Cornac interface
        self.num_users = None
        self.num_items = None
        self.uid_map = None
        self.iid_map = None
        
    def clone(self, params):
        new_model = HybridKNN75(**params)
        return new_model
        
    def fit(self, train_data, val_data=None):
        # Convert to CSR matrix for efficient operations
        self.train_matrix = train_data.csr_matrix
        self.num_users = train_data.num_users
        self.num_items = train_data.num_items
        # Copy the mapping attributes from train_data
        self.uid_map = train_data.uid_map
        self.iid_map = train_data.iid_map
        
        # Compute user-user similarity
        self.user_sim = cosine_similarity(self.train_matrix)
        np.fill_diagonal(self.user_sim, 0)  # Remove self-similarity
        
        # Compute item-item similarity  
        self.item_sim = cosine_similarity(self.train_matrix.T)
        np.fill_diagonal(self.item_sim, 0)  # Remove self-similarity
        
    def score(self, user_id):
        scores = np.zeros(self.num_items)
        
        # User-based prediction
        user_similarities = self.user_sim[user_id]
        top_users = np.argsort(user_similarities)[-self.k_users:]
        
        if len(top_users) > 0 and np.sum(np.abs(user_similarities[top_users])) > 1e-8:
            sim_weights = user_similarities[top_users]
            sim_weights = sim_weights / (np.sum(np.abs(sim_weights)) + 1e-8)
            user_scores = np.average(self.train_matrix[top_users].toarray(), 
                                   weights=np.abs(sim_weights), axis=0)
        else:
            user_scores = np.zeros(self.num_items)
            
        # Item-based prediction
        user_items = self.train_matrix[user_id].indices
        item_scores = np.zeros(self.num_items)
        
        if len(user_items) > 0:
            for item_id in range(self.num_items):
                item_similarities = self.item_sim[item_id, user_items]
                if len(item_similarities) > 0:
                    top_items_mask = np.argsort(item_similarities)[-min(self.k_items, len(item_similarities)):]
                    
                    if len(top_items_mask) > 0:
                        weights = item_similarities[top_items_mask]
                        if np.sum(np.abs(weights)) > 1e-8:  # Check if weights are non-zero
                            weights = weights / (np.sum(np.abs(weights)) + 1e-8)
                            item_scores[item_id] = np.average(
                                self.train_matrix[user_id, user_items[top_items_mask]].toarray().flatten(),
                                weights=np.abs(weights)
                            )
        
        # Hybrid combination
        scores = self.alpha * user_scores + (1 - self.alpha) * item_scores
        return scores

_hybrid_knn_75 = HybridKNN75(
    name="HybridKNN75",
    k_users=30,
    k_items=30, 
    alpha=0.75,
    seed=SEED,
)

_hybrid_knn_75_search_space = {
    "k_users": [30],
    "k_items": [30],
    "alpha": [0.75]
}
_hybrid_knn_75_search_space_tied_constraints = {}




# --------------------------------------------------------
import collections

if __name__ == "__main__":
    print("=" * 80)
    print("KNNBaseline Tourism Recommendation Example (Pure KNN-Baseline)")
    print("=" * 80)
    
    # Example: 5 tourists × 6 POIs
    user_item_matrix = np.array([
        [1, 0, 1, 0, 0, 1],  # Tourist A: visited Colosseum, Vatican, Trevi
        [0, 1, 1, 1, 0, 0],  # Tourist B: visited Pantheon, Vatican, Spanish Steps  
        [1, 1, 0, 0, 1, 1],  # Tourist C: visited Colosseum, Pantheon, Forum, Trevi
        [0, 0, 1, 1, 1, 0],  # Tourist D: visited Vatican, Spanish Steps, Forum
        [1, 0, 0, 0, 0, 1],  # Tourist E: visited Colosseum, Trevi
    ])
    
    poi_names = ["Colosseum", "Pantheon", "Vatican", "Spanish_Steps", "Roman_Forum", "Trevi_Fountain"]
    tourist_names = ["Tourist_A", "Tourist_B", "Tourist_C", "Tourist_D", "Tourist_E"]
    
    print("\nOriginal Visit Matrix:")
    print("Rows = Tourists, Columns = POIs")
    print("1 = Visited, 0 = Not Visited")
    print("-" * 50)
    
    # Print header
    print(f"{'Tourist':<12}", end="")
    for poi in poi_names:
        print(f"{poi:<15}", end="")
    print()
    
    # Print matrix with labels
    for i, tourist in enumerate(tourist_names):
        print(f"{tourist:<12}", end="")
        for j, visited in enumerate(user_item_matrix[i]):
            print(f"{visited:<15}", end="")
        print()
    
    # Create mappings for Cornac
    N, J = user_item_matrix.shape
    uid_map = collections.OrderedDict([(n, n) for n in range(N)])
    iid_map = collections.OrderedDict([(j, j) for j in range(J)])
    
    # Prepare data for Cornac
    train_data = prepare_positive_feedback_data(user_item_matrix, uid_map, iid_map)
    
    # Initialize and train KNNBaseline model
    print(f"\n{'='*50}")
    print("Training Pure KNNBaseline Model")
    print(f"{'='*50}")
    
    # UPDATED: Removed sustainability_boost parameter
    model = KNNBaseline(k=3, seed=SEED, verbose=True)  # Use k=3 for small example
    model.fit(train_data)
    
    # Display CORRECTED model statistics
    print(f"\nCorrected Model Statistics:")
    print(f"Global Mean (Fixed): {model.global_mean:.3f}")  # Now shows correct value (0.5)
    print(f"User Baselines: {[f'{x:.3f}' for x in model.user_baselines]}")
    print(f"Item Baselines: {[f'{x:.3f}' for x in model.item_baselines]}")
    
    # Show the correction impact
    print(f"\n🔍 Correction Impact:")
    print(f"  NEW Global Mean: {model.global_mean:.3f} (includes 0s)")
    print(f"  Result: More realistic baseline predictions!")
    
    # Calculate item popularity (no sustainability classification)
    item_counts = np.array(model.user_item_matrix.sum(axis=0)).flatten()
    print(f"\nPOI Popularity (visit counts):")
    for i, (poi, count) in enumerate(zip(poi_names, item_counts)):
        print(f"  {poi:<15}: {count} visits")
    
    # Show user similarities
    print(f"\nUser Similarity Matrix:")
    print("(How similar each tourist is to others based on visit patterns)")
    print("-" * 60)
    print(f"{'Tourist':<12}", end="")
    for tourist in tourist_names:
        print(f"{tourist:<12}", end="")
    print()
    
    for i, tourist in enumerate(tourist_names):
        print(f"{tourist:<12}", end="")
        for j in range(N):
            sim = model.user_similarities[i, j]
            print(f"{sim:<12.3f}", end="")
        print()
    
    # Make predictions for each tourist
    print(f"\n{'='*60}")
    print("Predictions for Each Tourist")
    print(f"{'='*60}")
    
    for user_idx in range(N):
        print(f"\n--- {tourist_names[user_idx]} ---")
        
        # Get predictions
        scores = model.score(user_idx)
        
        # UPDATED: Removed sustainability column
        print("POI Recommendations (Score | Visited):")
        
        # Sort by score for better visualization
        sorted_indices = np.argsort(scores)[::-1]
        
        for rank, poi_idx in enumerate(sorted_indices):
            poi_name = poi_names[poi_idx]
            score = scores[poi_idx]
            visited = "✓" if user_item_matrix[user_idx, poi_idx] == 1 else "✗"
            
            print(f"  {rank+1}. {poi_name:<15}: {score:.3f} | Visited: {visited}")
    
    # Demonstrate recommendation for a specific user
    print(f"\n{'='*60}")
    print("Detailed Recommendation Process for Tourist A")
    print(f"{'='*60}")
    
    user_idx = 0  # Tourist A
    print(f"\nTourist A visited: {[poi_names[i] for i, v in enumerate(user_item_matrix[user_idx]) if v == 1]}")
    
    # Find similar users
    similarities = model.user_similarities[user_idx]
    top_k_indices = np.argsort(similarities)[-model.k:]
    
    print(f"\nTop {model.k} most similar tourists:")
    for i, neighbor_idx in enumerate(top_k_indices):
        if similarities[neighbor_idx] > 1e-8:
            neighbor_visits = [poi_names[j] for j, v in enumerate(user_item_matrix[neighbor_idx]) if v == 1]
            print(f"  {i+1}. {tourist_names[neighbor_idx]} (similarity: {similarities[neighbor_idx]:.3f})")
            print(f"     Visited: {neighbor_visits}")
    
    # Show recommendation scores for unvisited POIs
    scores = model.score(user_idx)
    print(f"\nRecommendation scores for unvisited POIs:")
    for poi_idx, (poi_name, score) in enumerate(zip(poi_names, scores)):
        if user_item_matrix[user_idx, poi_idx] == 0:  # Not visited
            print(f"  {poi_name:<15}: {score:.3f}")
    
    # Show baseline vs KNN contribution
    print(f"\n📊 Score Breakdown for Tourist A:")
    baseline_scores = model.global_mean + model.user_baselines[user_idx] + model.item_baselines
    knn_scores = scores
    
    print(f"Global Mean: {model.global_mean:.3f}")
    print(f"User A Baseline: {model.user_baselines[user_idx]:.3f}")
    print(f"Item Baselines: {[f'{x:.3f}' for x in model.item_baselines]}")
    print(f"\nFor unvisited POIs:")
    for poi_idx, poi_name in enumerate(poi_names):
        if user_item_matrix[user_idx, poi_idx] == 0:
            baseline = baseline_scores[poi_idx]
            final = knn_scores[poi_idx]
            knn_adjustment = final - baseline
            print(f"  {poi_name:<15}: Baseline={baseline:.3f} + KNN={knn_adjustment:+.3f} = Final={final:.3f}")
    
    print(f"\n{'='*80}")
    print("Pure KNN-Baseline Example Complete!")
    print("The model successfully combines:")
    print("1. ✅ User similarity (collaborative filtering)")
    print("2. ✅ Baseline predictions (corrected user/item biases)")
    print("3. ✅ KNN adjustments (neighbor influence)")
    print("4. ❌ NO sustainability boost (pure algorithm)")
    print(f"{'='*80}")