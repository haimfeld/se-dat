"""Optional scikit-learn interoperability for :class:`sedat.EncodingPlan`.

``EncodingPlan.to_column_transformer()`` mirrors the suggested encodings with
native sklearn transformers:

- ``binary_encode``  -> OrdinalEncoder (alphabetical order reproduces the 0/1 map)
- ``one_hot``        -> OneHotEncoder(handle_unknown="ignore")
- ``ordinal_encode`` -> OrdinalEncoder(handle_unknown="use_encoded_value")
- ``target_encode``  -> TargetEncoder

Everything without a suggestion (already-numeric columns, ids, ...) flows
through ``remainder="passthrough"``. scikit-learn is imported lazily so it
stays an optional dependency; install it via ``pip install se-dat[sklearn]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .encoding import (
    ONE_HOT_STRATEGY,
    TARGET_STRATEGY,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .encoding import EncodingPlan


def _import_transformers():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import (
            OneHotEncoder,
            OrdinalEncoder,
            TargetEncoder,
        )
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "scikit-learn is required for to_column_transformer(). Install it with: pip install se-dat[sklearn]"
        ) from exc
    return ColumnTransformer, OneHotEncoder, OrdinalEncoder, TargetEncoder


def _sorted_categories(suggestion) -> list[str]:
    return sorted(str(value) for value in suggestion.categories)


def encoding_plan_to_column_transformer(plan: EncodingPlan):
    """Build an sklearn ``ColumnTransformer`` equivalent to ``plan``.

    The returned transformer is *ready to fit*: call
    ``transformer.fit(X, y)`` (``y`` is required when the plan contains
    target-encoding suggestions). Fitted on a training frame and applied to a
    test frame, unseen categories are handled gracefully instead of raising.
    """
    ColumnTransformer, OneHotEncoder, OrdinalEncoder, TargetEncoder = _import_transformers()

    transformers = []
    for suggestion in plan.suggestions:
        categories = [_sorted_categories(suggestion)]
        if suggestion.strategy == ONE_HOT_STRATEGY:
            estimator = OneHotEncoder(
                categories=categories,
                handle_unknown="ignore",
                sparse_output=False,
            )
        elif suggestion.strategy == TARGET_STRATEGY:
            estimator = TargetEncoder(categories="auto")
        else:
            # binary_encode: alphabetical order matches the stored {value: 0/1}
            # mapping for every supported binary value set (0/1, yes/no,
            # y/n, true/false, t/f); ordinal_encode uses the same ordering.
            estimator = OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        name = f"{suggestion.column}_{suggestion.strategy}"
        transformers.append((name, estimator, [suggestion.column]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
