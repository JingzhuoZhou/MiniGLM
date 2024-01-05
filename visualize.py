import matplotlib.pyplot as plt
import os
def visualize_loss(train_loss_list, train_interval, val_loss_list, val_interval, dataset, out_dir):
    ### visualize loss of training & validation and save to [out_dir]/loss.png
    
    
    # Create x-axis values based on train_interval and val_interval
    train_steps = [i * train_interval for i in range(len(train_loss_list))]
    valid_steps = [i * val_interval for i in range(len(val_loss_list))]

    # Create a figure and axis for the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss_list, label='Training Loss:'+str(round(train_loss_list[-1],2)), linestyle='-', marker='o', markersize=3)
    plt.plot(valid_steps, val_loss_list, label='Validation Loss:'+str(round(float(val_loss_list[-1]),2)), linestyle='-', marker='o', markersize=3)

    # Set plot labels and title
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Loss Visualization for {dataset} Dataset')

    # Add legend
    plt.legend()

    # Save the plot as an image file
    plt.savefig(os.path.join(out_dir, 'loss.png'))

    ###
    pass
