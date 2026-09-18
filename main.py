import random

def simple_decision_model(value, bias=0):
    """
    A very simple model to make a binary decision (0 or 1) based on a value and an optional bias.
    This simulates a basic classifier or decision maker.
    """
    # Decision is based on the input value adjusted by the bias.
    # If the combined score exceeds 0.5, it's a '1', otherwise '0'.
    return 1 if (value + bias) > 0.5 else 0

def autoregressive_decisions(input_values, initial_bias=0.0, dependency_strength=0.3):
    """
    Simulates an autoregressive decision process.
    Each decision depends on the current input and the *previous decision's influence*.
    This creates a sequential dependency, where past outputs affect future ones.
    """
    print("\n--- Autoregressive Decisions ---")
    decisions = []
    # `current_bias` represents the cumulative influence of previous decisions.
    # In a real AR model, this might be a hidden state or a direct input of the previous output.
    current_bias = initial_bias
    for i, value in enumerate(input_values):
        # Make a decision using the current input value and the accumulated bias from previous steps.
        # This is where the dependency on previous outputs is simulated.
        decision = simple_decision_model(value, current_bias)
        decisions.append(decision)
        print(f"Input[{i}]: {value:.2f}, Prev Decision Influence: {current_bias:.2f} -> Decision: {decision}")

        # Update the bias for the *next* decision based on the *current* decision.
        # This is the core autoregressive dependency mechanism.
        if decision == 1:
            current_bias += dependency_strength
        else:
            current_bias -= dependency_strength
        # Clamp bias to keep it within a reasonable range for demonstration.
        current_bias = max(-0.5, min(0.5, current_bias))
    return decisions

def non_autoregressive_decisions(input_values):
    """
    Simulates a non-autoregressive decision process.
    Each decision is made independently based *only* on the current input.
    There is no dependency on previously generated outputs within the same prediction pass.
    This allows for potential parallelization of decisions.
    """
    print("\n--- Non-Autoregressive Decisions ---")
    decisions = []
    for i, value in enumerate(input_values):
        # Make a decision based solely on the current input value.
        # No 'bias' or influence from previous decisions in this sequence.
        decision = simple_decision_model(value)
        decisions.append(decision)
        print(f"Input[{i}]: {value:.2f} -> Decision: {decision}")
    return decisions

# --- Main execution ---
random.seed(42) # For reproducibility of random inputs
# Generate a sequence of sample input values (e.g., sensor readings, stock indicators)
sample_inputs = [random.random() for _ in range(10)]

print("--- Comparing Autoregressive vs. Non-Autoregressive Decision Models ---")
print("Sample Input Values:", [f"{x:.2f}" for x in sample_inputs])

# Run the autoregressive simulation
ar_results = autoregressive_decisions(sample_inputs)

# Run the non-autoregressive simulation
nar_results = non_autoregressive_decisions(sample_inputs)

print("\n--- Summary ---")
print("Autoregressive Final Decisions:", ar_results)
print("Non-Autoregressive Final Decisions:", nar_results)
print("\nNotice how the autoregressive model's decisions can 'drift' based on previous outputs,")
print("while the non-autoregressive model makes each decision independently.")
