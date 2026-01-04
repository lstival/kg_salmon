def explain_models():
    """
    Returns an explanation of the selected KG encoder models and their differences.
    """
    explanations = {
        "TransE": (
            "TransE (Translational Embeddings) represents relations as translations in the embedding space. "
            "It assumes that for a triple (h, r, t), the embedding of the tail 't' should be close to h + r. "
            "It is effective for simple hierarchical or 1-to-1 relationships but struggles with complex relations like 1-to-N or N-to-N."
        ),
        "AutoSF": (
            "AutoSF is an Automated Search framework for structural Scoring Functions. "
            "Instead of a fixed mathematical form, it searches for the optimal scoring function for a given dataset "
            "by defining it as a sum of products between head, relation, and tail representation components. "
            "It is highly flexible and often outperforms human-designed models."
        ),
        "RotatE": (
            "RotatE models relations as rotations in a complex vector space. "
            "For a triple (h, r, t), it expects t = h ⊙ r, where |r_i| = 1. "
            "This allows it to capture various relation patterns like symmetry, antisymmetry, inversion, and composition."
        ),
        "DistMult": (
            "DistMult is a bilinear model that uses element-wise multiplication. "
            "The score is calculated as a trilinear product <h, diag(r), t>. "
            "Since the score is symmetric, it is good for capturing symmetric relations but cannot model directed ones."
        ),
        "ComplEx": (
            "ComplEx (Complex Embeddings) extends DistMult by using complex-valued embeddings. "
            "It can capture both symmetric and antisymmetric relations, making it more expressive than DistMult "
            "while maintaining a linear time and space complexity."
        )
    }
    return explanations
