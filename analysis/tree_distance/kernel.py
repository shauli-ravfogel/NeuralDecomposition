from functools import reduce
from typing import Sequence, Union, Tuple, Dict, List, Iterable, Callable, Optional
from typing_extensions import Protocol
import utils as U
import numpy as np

class Labeled(Protocol):
    def label(self) -> str: ...

class TreeLike(Sequence['TreeLike'], Labeled): ...

# These functions work with nltk.tree.Tree

def label(n: Union[Labeled, str]) -> str:
    if isinstance(n, str):
        return n
    else:
        return n.label()

def children(n: TreeLike) -> Sequence[TreeLike]:
    if isinstance(n, str):
        return []
    else:
        return n[:]

def delex(n, leaf=""):
    from nltk.tree import Tree
    if isinstance(n, str): 
        return leaf 
    else: 
        return Tree(n.label(), [ delex(c) for c in n[:] ])

class Kernel:
    """Class to hold configuration of the kernel function."""
    def __init__(self,
                 label: Callable[[Union[Labeled, str]], str]        =label,
                 children: Callable[[TreeLike], Sequence[TreeLike]] =children,
                 alpha: float                                       =1.0):
        self.label = label
        self.children = children
        self.alpha = alpha

    def leaf(self, t: TreeLike) -> bool:
        return len(self.children(t)) == 0
    
    def subtrees(self, t: TreeLike) -> Iterable[TreeLike]:
        """Yields all subtrees of a tree t."""
        if self.leaf(t):
            pass
        else:
            yield t
            for c in self.children(t):
                yield from self.subtrees(c)

    def preterm(self, t: TreeLike) -> bool:
        """Returns True if node t is a pre-terminal."""
        return not(self.leaf(t)) and all((self.leaf(c) for c in self.children(t)))
        
    
    def production(self, t: TreeLike) -> tuple:
        """Returns the productiona at node t, i.e. the list of children's labels."""
        return tuple(self.label(c) for c in self.children(t))

    def C(self, n1: TreeLike, n2: TreeLike) -> float:
        # both nodes are preterminals and have same productions
        if self.preterm(n1) and self.preterm(n2) and self.label(n1) == self.label(n2) and (self.production(n1) == self.production(n2)):
            return self.alpha
        # both nodes are non-terminals and have same productions
        elif not self.preterm(n1) and not self.preterm(n2) and self.label(n1) == self.label(n2) and self.production(n1) == self.production(n2):
            return self.alpha * product(1 + self.C(self.children(n1)[i], self.children(n2)[i]) for i in range(len(self.children(n1))))
        else:
            return 0

    def __call__(self, t1: TreeLike, t2: TreeLike) -> float:
        """Returns the number of shared tree fragments between trees t1 and t2, discounted by alpha."""
        N = sum(self.C(n1, n2) for n1 in self.subtrees(t1) for n2 in self.subtrees(t2))
        return N

    def ftk(self, nodes1: Dict[tuple, List[TreeLike]] , nodes2: Dict[tuple, List[TreeLike]]) -> float:
        """Returns the number of shared tree fragments between nodemaps nodes1
        and nodes2, discounted by alpha.  Algorithm adapted from:
        Moschitti, A. (2006). Making tree kernels practical for
        natural language learning. In 11th conference of the European
        Chapter of the Association for Computational
        Linguistics. http://www.aclweb.org/anthology/E06-1015
        """
        
        ks1 = nodes1.keys()
        ks2 = nodes2.keys()
        N = sum(self.C(n1, n2)
                for k in set(ks1).union(ks2)
                  for n1 in nodes1.get(k, [])
                    for n2 in nodes2.get(k, []))
        return N

    def pairwise(self,
                 trees1: Sequence[TreeLike],
                 trees2: Optional[Sequence[TreeLike]]=None,
                 normalize: bool=False,
                 dtype: type=np.float64):
        """Returns the value of the tree kernel between sequence of trees1 and trees2, 
        using the Fast Tree Kernel algorithm of Moschitti, A. (2006). 
        """
        nodes1 = [ self.nodemap(t) for t in trees1 ]
        if trees2 is not None:
            nodes2: Optional[List[Dict[tuple, List[TreeLike]]]] = [ self.nodemap(t) for t in trees2 ]
        else:
            nodes2 = None
        # For some reason this doesn't parallelize well: we'll call the sequential version of U.pairwise
        return U.pairwise(self.ftk, nodes1, data2=nodes2, normalize=normalize, dtype=dtype, parallel=False)
    
    def nodemap(self, t: TreeLike) -> Dict[tuple, List[TreeLike]]:
        """Returns the map from productions to lists of nodes for given tree t."""
        M: Dict[tuple, List[TreeLike]] = {}
        for n in self.subtrees(t):
            p = self.production(n)
            if p in M:
                M[p].append(n)
            else:
                M[p] = [n]
        return M
    
def product(xs: Iterable[float]) -> float:
    return reduce(lambda a, b: a*b, xs, 1.0)
