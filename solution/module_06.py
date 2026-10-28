"""Frozen auxiliary forcing with owned state and independent layer batching.

Optional per-layer table (Claude, 18 Oct): when data_03.npz carries 'layers' (one 204-vector per layer 1..depth-2), the forcing
window starts at layer 1 and each layer uses its own row; otherwise the shipped middle/late window (depth//2 ..) is unchanged."""
from pathlib import Path
import flopscope.numpy as f
from module_02 import reserve, response as graph_response
from module_05 import mean_gain


def install(cls, policy):
    if policy not in ('middle', 'late', 'pooled', 'separate'):
        raise ValueError(policy)
    setup0 = cls.setup
    observe0 = cls._higher_observe
    finish0 = cls._finish_response
    cls.LATE_RESPONSE_ENABLED = True

    def setup(self, ctx):
        setup0(self, ctx)
        model = f.load(str(Path(__file__).with_name('data_03.npz')))
        self._late_theta_middle = model['middle'].astype(f.float32)
        self._late_theta_late = model['late'].astype(f.float32)
        self._late_theta_layers = model['layers'].astype(f.float32) if 'layers' in model else None
        self._late_workspace = {}
        controller = self._fold_controller
        birth0 = controller.observe_birth

        def birth(li, d21, g4):
            self._late_d21 = d21
            return birth0(li, d21, g4)
        controller.observe_birth = birth

    def observe(self, W, mu, a, s, phi, g, d3, g4, cov, sketch,
                li, depth, inverse, wick):
        result = observe0(self, W, mu, a, s, phi, g, d3, g4, cov,
                          sketch, li, depth, inverse, wick)
        if li == 0:
            self._late_records = []
            self._late_split = (3*depth)//4
            self._late_start = self._late_split if policy == 'late' else depth//2
            self._late_stop = self._late_split if policy == 'middle' else depth-1
            table = self._late_theta_layers
            per_layer = table is not None and table.shape[0] == depth-2
            if per_layer:
                self._late_start = 1
                self._late_stop = depth-1
            self._late_enabled = (self.LATE_RESPONSE_ENABLED and a.shape[0] >= 32
                                  and self._late_start < self._late_stop)
            if self._late_enabled:
                count = self._late_stop-self._late_start
                reserve(self._late_workspace, count, a.shape[0])
                theta_key = (depth, policy, per_layer)
                if self._late_workspace.get('theta_key') != theta_key:
                    self._late_workspace['theta'] = (f.copy(table[self._late_start-1:self._late_stop-1]) if per_layer else f.stack(tuple(
                        self._late_theta_middle if j < self._late_split else self._late_theta_late
                        for j in range(self._late_start, self._late_stop))))
                    self._late_workspace['theta_key'] = theta_key
        if not self._late_enabled or li < self._late_start:
            return result
        # Retain exact callback alpha/sigma, and own all mutable input storage.
        self._late_records.append(f.stack((mu, s*s, d3, g4, a, s), axis=0))
        if li < self._late_stop:
            j = li-self._late_start
            f.copyto(self._late_workspace['R'][j], cov)
            f.copyto(self._late_workspace['T'][j], self._late_d21)
        self._late_d21 = None
        return result

    def finish(self, base, mlp):
        if not self._late_enabled:
            return finish0(self, base, mlp)
        records = f.stack(self._late_records)
        full_inverse = f.stack(self._higher_inverses, axis=0)
        self._higher_inverse_stack = full_inverse
        inverse = full_inverse[self._late_start-1:]
        count = self._late_stop-self._late_start
        ws = self._late_workspace
        values = graph_response(records[:count, 4], records[:count, 5],
                                records[:count, 2], records[:count, 3],
                                ws['R'], ws['T'], ws['theta'], ws)
        forcing = f.take_along_axis(values, inverse[:count], axis=1)
        marginals = records[:, :4].astype(f.float64)
        gains = mean_gain(marginals[:, 0], marginals[:, 1], marginals[:, 2], marginals[:, 3])
        gains = f.take_along_axis(gains, inverse, axis=1)
        output = finish0(self, base, mlp)
        f.maximum(output, 0., out=output)
        delta = f.zeros(mlp.width, dtype=f.float64)
        start = self._late_start
        wt = self._readout_weight_transposes
        deltas = []
        for li in range(start, mlp.depth):
            if li > start:
                delta = gains[li-start]*(wt[li]@delta)
            if li < self._late_stop:
                delta = delta+forcing[li-start]
            deltas.append(delta)
        f.copyto(output[start:], (output[start:].astype(f.float64)+f.stack(deltas, axis=0)).astype(f.float32))
        self._late_records = []
        self._late_d21 = None
        return output

    cls.setup = setup
    cls._higher_observe = observe
    cls._finish_response = finish
