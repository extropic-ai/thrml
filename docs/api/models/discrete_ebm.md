# Discrete Energy-Based Models

This module contains implementations of discrete energy-based models.

::: thrml.models.DiscreteEBMFactor
    options:
        members:
            - energy
            - to_interaction_groups

::: thrml.models.DiscreteEBMInteraction
    options:
        members: False

::: thrml.models.SquareDiscreteEBMFactor
    options:
        members:
            - to_interaction_groups

::: thrml.models.SpinEBMFactor
    options:
        members: false
        inherited_members: false

::: thrml.models.CategoricalEBMFactor
    options:
        members: false
        inherited_members: false

::: thrml.models.SquareCategoricalEBMFactor
    options:
        members: false
        inherited_members: false

::: thrml.models.SpinGibbsConditional
    options:
        members:
            - compute_parameters

::: thrml.models.CategoricalGibbsConditional
    options:
        members:
            - compute_parameters