
def get_alpha(confidence, alternative):
    if confidence <= 0 or confidence >= 1:
        raise ValueError()
    if alternative not in ("one-sided", "two-sided"):
        raise ValueError()
        
    alpha = 1 - confidence
    
    if alternative == "two-sided":
        return alpha / 2
    return alpha
