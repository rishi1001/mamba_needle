import matplotlib.pyplot as plt

# Load losses from text files
with open("seq_train_losses.txt", "r") as f:
    seq_train_losses = [float(line.strip()) for line in f]

with open("pscan_train_losses.txt", "r") as f:
    pscan_train_losses = [float(line.strip()) for line in f]

# Plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(seq_train_losses, label="Sequential Loss", color="blue")
plt.plot(pscan_train_losses, label="Parallel Scan Loss", color="orange")
plt.title("Training Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save or display the plot
plt.savefig("loss_comparison_plot.png")  # Saves the plot as a PNG file
# plt.show()  # Displays the plot
