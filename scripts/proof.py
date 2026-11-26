import numpy as np
import time
LABELS=['=','>','<','~']
import collections

from scipy.sparse import coo_matrix

from scipy.sparse.csgraph import connected_components

BITS=(1,2,4,8)

FORMAL_SCOPE='atomic strict4 workingGold + weak subset; potential proof audit, not inferredGold'

def labels(mask):
    return "|".join(r for r, bit in zip(LABELS, BITS) if mask & bit)

def mask_from_info(rab, rba, nsab, nsba):
    """Exact related-query feasibility only in a consistent declared fragment."""
    return (int(not nsab and not nsba)
            + 2 * int(not rab and not nsba)
            + 4 * int(not rba and not nsab)
            + 8 * int(not rab and not rba))

def potential_from_info(rab, rba, nsab, nsba):
    """Monotone sufficient proof information; not uniqueness on conflicts."""
    return (int(rab and rba)
            + 2 * int(rba and nsab)
            + 4 * int(rab and nsba)
            + 8 * int(nsab and nsba))

class Network:
    def __init__(self, nodes, weak_edges, anchors):
        started = time.perf_counter()
        self.nodes = list(nodes)
        self.index = {v: i for i, v in enumerate(self.nodes)}
        self.anchors = list(anchors)  # source, target, atomic relation, TRAIN row id
        n = len(self.nodes)
        edge_set = set()
        self.edge_origin = {}
        self.obligations = []
        for ei, (a, b) in enumerate(weak_edges):
            u, v = self.index[a], self.index[b]
            edge_set.add((u, v))
            self.edge_origin.setdefault((u, v), {"type": "public_weak_subClassOf", "edge_index": ei})
        for a, b, relation, row in self.anchors:
            u, v = self.index[a], self.index[b]
            adds = []
            if relation in ("=", "<"):
                adds.append((u, v))
            if relation in ("=", ">"):
                adds.append((v, u))
            for e in adds:
                edge_set.add(e)
                self.edge_origin.setdefault(e, {"type": "train_Gold_inclusion_consequence", "train_row": int(row), "label": relation})
            if relation in (">", "~"):
                self.obligations.append((u, v, int(row), "source_only"))
            if relation in ("<", "~"):
                self.obligations.append((v, u, int(row), "target_only"))
        self.edges = sorted(edge_set)
        if self.edges:
            ii, jj = zip(*self.edges)
            mat = coo_matrix((np.ones(len(ii), dtype=np.int8), (ii, jj)), shape=(n, n)).tocsr()
        else:
            mat = coo_matrix((n, n)).tocsr()
        nc, self.component = connected_components(mat, directed=True, connection="strong")
        self.n_components = int(nc)
        parents = [set() for _ in range(nc)]
        children = [set() for _ in range(nc)]
        for u, v in self.edges:
            a, b = int(self.component[u]), int(self.component[v])
            if a != b:
                parents[a].add(b)
                children[b].add(a)
        remaining = [len(p) for p in parents]
        queue = collections.deque(i for i, k in enumerate(remaining) if k == 0)
        self.reach = [1 << i for i in range(nc)]
        order = []
        while queue:
            p = queue.popleft()
            order.append(p)
            for child in children[p]:
                self.reach[child] |= self.reach[p]
                remaining[child] -= 1
                if not remaining[child]:
                    queue.append(child)
        assert len(order) == nc
        full = [0] * nc
        good = [0] * nc
        self.violations = []
        for u, v, row, side in self.obligations:
            cu, cv = int(self.component[u]), int(self.component[v])
            full[cu] |= 1 << cv
            if self.reach[cu] & (1 << cv):
                self.violations.append({"train_row": row, "side": side, "positive": self.nodes[u], "negative": self.nodes[v]})
            else:
                good[cu] |= 1 << cv
        for child in reversed(order):
            for p in parents[child]:
                full[p] |= full[child]
                good[p] |= good[child]
        self.full_not_subset = full
        self.good_not_subset = good
        self.stats = {
            "nodes": n, "inclusion_edges": len(self.edges), "inclusion_SCCs": int(nc),
            "train_anchors": len(self.anchors), "difference_obligations": len(self.obligations),
            "violated_difference_obligations": len(self.violations),
            "full_fit_consistent": not self.violations,
            "weakening_dropped_obligations": len(self.violations),
            "build_seconds": time.perf_counter() - started,
            "scope": FORMAL_SCOPE,
        }

    def query(self, source, target):
        a = int(self.component[self.index[source]])
        b = int(self.component[self.index[target]])
        rab = bool(self.reach[a] & (1 << b))
        rba = bool(self.reach[b] & (1 << a))
        nsab = bool(self.full_not_subset[a] & self.reach[b])
        nsba = bool(self.full_not_subset[b] & self.reach[a])
        gab = bool(self.good_not_subset[a] & self.reach[b])
        gba = bool(self.good_not_subset[b] & self.reach[a])
        potential = potential_from_info(rab, rba, nsab, nsba)
        weakened = mask_from_info(rab, rba, gab, gba)
        exact = 0 if self.violations else mask_from_info(rab, rba, nsab, nsba)
        assert weakened != 0
        if weakened.bit_count() == 1:
            assert potential & weakened
        if exact.bit_count() == 1:
            assert potential & exact
        return {
            "RAB": rab, "RBA": rba, "NSAB": nsab, "NSBA": nsba,
            "potential_mask": potential, "potential_labels": labels(potential),
            "potential_proof_conflict": potential.bit_count() > 1,
            "full_exact_mask": exact, "weakening_exact_mask": weakened,
            "weakening_exact_labels": labels(weakened),
            "block": potential != 0,
        }

    def _path(self, first, second):
        a, b = self.index[first], self.index[second]
        if a == b:
            return []
        adj = collections.defaultdict(list)
        for u, v in self.edges:
            adj[u].append(v)
        previous = {a: None}
        queue = collections.deque([a])
        while queue and b not in previous:
            u = queue.popleft()
            for v in adj[u]:
                if v not in previous:
                    previous[v] = u
                    queue.append(v)
        assert b in previous
        seq = []
        v = b
        while v != a:
            u = previous[v]
            seq.append({"source": self.nodes[u], "target": self.nodes[v], **self.edge_origin[(u, v)]})
            v = u
        return list(reversed(seq))

    def certificate(self, source, target, kind):
        def inclusion(a, b):
            return {"claim": [a, "weak_subset", b], "path": self._path(a, b)}

        def difference(a, b):
            ai, bi = self.index[a], self.index[b]
            ca, cb = int(self.component[ai]), int(self.component[bi])
            for u, v, row, side in self.obligations:
                cu, cv = int(self.component[u]), int(self.component[v])
                if self.reach[cu] & (1 << ca) and self.reach[cb] & (1 << cv):
                    return {"claim": [a, "not_subset", b], "train_row": row, "side": side,
                            "difference_positive": self.nodes[u], "difference_negative": self.nodes[v],
                            "positive_to_query": self._path(self.nodes[u], a),
                            "query_to_negative": self._path(b, self.nodes[v])}
            raise AssertionError("NS proof reconstruction failed")
        if kind == "=":
            parts = [inclusion(source, target), inclusion(target, source)]
        elif kind == ">":
            parts = [inclusion(target, source), difference(source, target)]
        elif kind == "<":
            parts = [inclusion(source, target), difference(target, source)]
        else:
            parts = [difference(source, target), difference(target, source)]
        return {"potential_relation": kind, "premise": "supplied query is related/nonempty intersection",
                "parts": parts, "claim_limit": "potential obligation-level proof only; may coexist with conflicting proofs; never new Gold"}

