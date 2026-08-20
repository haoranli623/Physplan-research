# Ranking Diagnostic Summary

## MAIN RESULT

**PREDICTOR NOT BOTTLENECK** on 120 fresh cloned states and 41 identical candidates per state.

## GT-LATENT RANKING QUALITY

Mean/median rho 0.720/0.797; mean regret 0.0273.

## PREDICTED-LATENT RANKING QUALITY

Mean/median rho 0.708/0.796; mean regret 0.0311.

## ORACLE GAP

Delta rho 0.012 [-0.034, 0.058]; delta regret 0.0038 [-0.0059, 0.0150].

## BOUNDARY-SPECIFIC EVIDENCE

Near-minus-far excess degradation -0.055 [-0.143, 0.032].

## WHAT THIS RULES OUT

It rules out a material GT-to-predicted ranking loss and boundary-specific ranking degradation at the frozen effect-size thresholds. It does not prove the predictor is universally sufficient outside this H6 local slice.

## WHAT THIS SUPPORTS

Decision-useful information is recoverable from true latents, and the standard JEPA predictor preserves nearly all of it here. The earlier high planner floor came from the fixed latent-goal scoring interface.

## NEXT RESEARCH STEP

Do not train Boundary-JEPA for this protocol. If continuing, study a better task-conditioned latent scoring/readout interface or test a prospectively chosen task where predictor degradation—not scorer mismatch—is independently demonstrated.
