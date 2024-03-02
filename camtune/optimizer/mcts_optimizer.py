import os 
import torch
from torch.quasirandom import SobolEngine
from typing import Callable, List, Dict, Tuple, Any
from botorch.utils.transforms import unnormalize

from camtune.utils import print_log, DEVICE, DTYPE

from .mcts_utils import Node
from .base_optimizer import BaseOptimizer
from .gp_bo_optimizer import GPBOOptimizer
from .turbo_optimizer import TuRBO
from .optim_utils import assign_after_check, round_by_bounds


OPTIMIZER_MAP = {
    'gp_bo': GPBOOptimizer,
    'turbo': TuRBO,
}
MCTS_ATTRS = {
    'Cp': 1.,
    'leaf_size': 2,
    'node_selection_type': 'UCB',
    'initial_sampling_method': 'Sobol',
    'save_path': False,

    'local_optimizer_type': 'turbo',
    'local_optimizer_params':  {
        'acqf': "ts",
    },

    'classifier_type': 'svm',
    'classifier_params': {
        'kernel_type': 'rbf',
        'gamma_type': 'auto',
    }
}
MIX_BETA = True

class MCTS(BaseOptimizer):
    def __init__(
        self,
        bounds: torch.Tensor, # shape: (2, dimension)
        obj_func: Callable,
        batch_size: int = 1,
        seed: int = 0,
        discrete_dims: List[int] = [],
        optimizer_params: Dict[str, Any] = None,
    ):

        self.seed = seed
        torch.manual_seed(self.seed)

        self.batch_size = batch_size
        self.bounds = bounds
        self.dimension = bounds.shape[1]
        self.discrete_dims = discrete_dims
        self.obj_func = obj_func

        for k, v in MCTS_ATTRS.items():
            if optimizer_params is None or k not in optimizer_params:
                setattr(self, k, v)
            else:
                setattr(self, k, optimizer_params[k])

        self.global_num_init: int = 10 if 'global_num_init' not in optimizer_params else optimizer_params['global_num_init']
        self.local_num_init: int = self.global_num_init if 'local_num_init' not in optimizer_params else optimizer_params['local_num_init']
        self.local_optimizer_params['n_init'] = self.local_num_init
        self.optimizer_cls = OPTIMIZER_MAP[self.local_optimizer_type.lower()]

        self.num_calls: int = 0
        self.X = torch.empty((0, self.dimension), dtype=DTYPE, device=DEVICE)
        self.Y = torch.empty((0, 1), dtype=DTYPE, device=DEVICE)

        self.best_X: torch.Tensor = None
        self.best_Y: torch.Tensor = None
    
    def initial_sampling(self, num_init: int):
        print_log(f'[MCTS] Initial Sampling: {num_init} samples', print_msg=True)

        if self.initial_sampling_method == 'Sobol':
            sobol = SobolEngine(dimension=self.dimension, scramble=True, seed=self.seed)
            X_init = sobol.draw(n=num_init).to(dtype=DTYPE, device=DEVICE)

            if MIX_BETA:
                X_init = unnormalize(X_init, self.bounds)
                X_init[:, self.discrete_dims] = round_by_bounds(X_init[:, self.discrete_dims], self.bounds[:, self.discrete_dims])

            Y_init = torch.tensor(
                [self.obj_func(x) for x in X_init], dtype=DTYPE, device=DEVICE,
            ).unsqueeze(-1)
        else:
            raise ValueError(f"Initial sampling method {self.initial_sampling_method} not supported")
    
        self.X = torch.cat([self.X, X_init], dim=0)
        self.Y = torch.cat([self.Y, Y_init], dim=0)
        self.root = self.init_node(parent=None, X=self.X, Y=self.Y, label=None, is_pos=None)

        self.global_num_init: int = num_init
        self.local_num_init: int = num_init # Currently the local optimizer's num_init is the same as the global one
        self.num_calls += num_init

    def clear_tree(self):
        curr_node: 'Node' = self.root
        node_queue: List['Node'] = [curr_node]
        while len(node_queue) > 0:
            curr_node = node_queue.pop(0)
            node_queue.extend(curr_node.children)

            Node.obj_counter -= 1
            del curr_node

        self.root = self.init_node(parent=None, X=self.X, Y=self.Y, label=None, is_pos=None)

    def init_node(self, parent: Node, X: torch.Tensor, Y: torch.Tensor, label: int, is_pos: bool):
        node = Node(
            parent=parent,
            label=label,
            bounds=self.bounds,
            classifier_type=self.classifier_type,
            classifier_params=self.classifier_params,
            seed=self.seed,
        )
        node.update_sample_bag(X, Y)

        # Checking label assignment
        if parent is not None:
            child_label: int = node.label
            target_label: int = parent.classifier.pos_label if is_pos else 1 - parent.classifier.pos_label
            if child_label != target_label:
                raise ValueError(f"Node {node.id} with {node.sample_bag[0].shape[0]} got unexpected label {child_label} "
                                 f" (expceted {target_label}, where is_pos={is_pos})")

            target_child_data: torch.Tensor = node.sample_bag[0]
            labels = parent.classifier.predict(target_child_data)
            if not (labels == child_label).all():
                raise ValueError(f"Node {node.id} with {node.sample_bag[0].shape[0]} samples is not "
                                 f"classified to label {child_label} by its parent's classifier. "
                                 f"Got labels: {labels}")

        return node

    def build_tree(self):
        self.clear_tree()

        nodes_splittable: List['Node'] = [self.root]
        while len(nodes_splittable) > 0:
            nodes_to_split = [node for node in nodes_splittable]
            nodes_splittable = []
            for node in nodes_to_split:
                try:
                    # pos_data - ((num_pos_samples, dimension), (num_pos_samples, 1))
                    # neg_data - ((num_neg_samples, dimension), (num_neg_samples, 1))
                    pos_data, neg_data, pos_label = node.fit()
                except:
                    continue

                if pos_data[0].shape[0] == 0 or neg_data[0].shape[0] == 0:
                    continue

                if pos_data[0].shape[0] > 0:
                    # pos_child's data should be classified to pos_label by its parent's classifier
                    pos_child: Node = self.init_node(node, pos_data[0], pos_data[1], label=pos_label, is_pos=True)
                    if pos_child.num_visits > self.leaf_size:
                        nodes_splittable.append(pos_child)
    
                if neg_data[0].shape[0] > 0:
                    neg_child: Node = self.init_node(node, neg_data[0], neg_data[1], label=1 - pos_label, is_pos=False)
                    if neg_child.num_visits > self.leaf_size:
                        nodes_splittable.append(neg_child)
    
                print_log(f'[MCTS] Build Tree: Split onto {pos_data[0].shape[0]} positive samples (node: {pos_child.id})'
                      f' and {neg_data[0].shape[0]} negative samples (node: {neg_child.id})'
                      f' where pos label is {pos_label}', print_msg=True)


    def get_node_score(self, node: Node) -> float:
        if self.node_selection_type == 'UCB':
            return node.get_UCB(self.Cp)
        else:
            raise ValueError(f"Node selection type {self.node_selection_type} not supported")

    def select(self) -> List['Node']:
        curr_node: Node = self.root

        path: List[Node] = [curr_node]
        while len(curr_node.children) > 0:
            curr_node = max(curr_node.children, 
                            key=lambda node: self.get_node_score(node))
            path.append(curr_node)
        
        Node.check_path(path)
        if self.save_path:
            self.save_path_info(path)
        return path

    def local_modelling(
            self, 
            num_evals: int, 
            path: List['Node'], 
            bounds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            path: List[Node] - A list of nodes from the root to the current node
            bounds: torch.Tensor - (2, dimension) - The bounds of the search space
        """
        optimizer: BaseOptimizer = self.optimizer_cls(
            bounds=bounds,
            batch_size=self.batch_size,
            obj_func=self.obj_func,
            seed=self.seed,
            discrete_dims=self.discrete_dims,
            in_mcts=True,
            optimizer_params=self.local_optimizer_params,
        )
        leaf_node = path[-1]

        mcts_params = {
            'n_init': self.local_num_init,
            'path': path,
            'X_in_region': leaf_node.sample_bag[0],
            'Y_in_region': leaf_node.sample_bag[1],
        }
        return optimizer.optimize(
            num_evals=num_evals, 
            mcts_params=mcts_params,
        )
    
    def optimize(self, num_evals: int, X_init: torch.Tensor, Y_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.X = torch.cat([self.X, X_init], dim=0)
        self.Y = torch.cat([self.Y, Y_init], dim=0)
        self.root = self.init_node(parent=None, X=self.X, Y=self.Y, label=None, is_pos=None)

        num_init = X_init.shape[0]
        self.global_num_init: int = num_init
        self.local_num_init: int = num_init # Currently the local optimizer's num_init is the same as the global one

        # Assume num_evals has already excluded the initial samples
        self.optimize(num_evals)

    def optimize(self, num_evals: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.initial_sampling(self.global_num_init)

        while self.num_calls < num_evals:
            print_log('-' * 60, print_msg=True)
            print_log(f'[MCTS] optimize: Building tree from call: {self.num_calls}', print_msg=True)
            self.build_tree()
            print_log('-' * 30)

            path: List['Node'] = self.select()
            print_log('Selected Path: ')
            for node in path:
                print_log(f'Node {node.id} with {node.sample_bag[0].shape[0]} samples ->', end=' ', print_msg=True)
            print_log('\n')
            print_log('-' * 30)

            cands, cand_vals = self.local_modelling(
                num_evals=num_evals - self.num_calls,
                path=path, 
                bounds=self.bounds
            )
            print_log('-' * 30)

            self.X = torch.cat([self.X, cands], dim=0)
            self.Y = torch.cat([self.Y, cand_vals], dim=0)
            self.num_calls += cands.shape[0]      

            if self.best_Y is None or cand_vals.max() > self.best_Y:
                self.best_X = cands[cand_vals.argmax()]
                self.best_Y = cand_vals.max()

            print_log(f'[MCTS] optimize: Best value found: {self.best_Y:.3e} after '
                      f'{self.num_calls} calls in total', print_msg=True)
            print_log('-' * 60, print_msg=True)

        return self.X, self.Y

    # def save_path_info(self, path: List['Node']):
    #     if len(path) < 2:
    #         return

    #     svm_save_dir = os.path.join(PATH_DIR, get_expr_name())
    #     if not os.path.exists(svm_save_dir): os.makedirs(svm_save_dir)
    #     for i in range(len(path) - 1):
    #         node, child = path[i], path[i+1]
    #         pkl_path: str = os.path.join(svm_save_dir, f'{self.num_calls}_{node.id}_.pkl')
    #         node.save_classifier(pkl_path)

    #         label_path: str = os.path.join(svm_save_dir, f'{self.num_calls}_{node.id}_.txt')
    #         with open(label_path, 'w') as f:
    #             f.write(f'{child.label}')

    #     last_parent: Node = path[-2]
    #     plot_path = os.path.join(svm_save_dir, f'{self.num_calls}_{last_parent.id}_.png')
    #     last_parent.plot_node_region(plot_path)