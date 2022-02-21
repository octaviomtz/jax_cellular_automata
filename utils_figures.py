import matplotlib.pyplot as plt

def fig_loss_and_synthesis(imgs_syn, losses, save=False, label='loss'):
    fig = plt.figure(figsize=(24,8))
    gs = fig.add_gridspec(4,11)
    count=0
    for r in range(4):
        for c in range(8):
            ax_ = fig.add_subplot(gs[r, c+3])
            ax_.imshow(imgs_syn[count])
            ax_.axis('off')
            count += 2
    ax2 = fig.add_subplot(gs[:4, :4])
    ax2.semilogy(losses, label=label)
    ax2.legend(fontsize=16)
    ax2.set_xlabel('epochs', fontsize=16)
    ax2.set_xlabel('MSE', fontsize=16)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    if save:
        plt.savefig(save)