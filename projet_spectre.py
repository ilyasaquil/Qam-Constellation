import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import math


def generate_qam_constellation(M, exclude_points):
    """
    Generate QAM constellation points, excluding specific corner points.         - excluded_points (ndarray): The excluded points.
    """
    side_len = int(np.sqrt(M))
    # Adjust side lengths for specific QAM sizes
    side_lengths = {32: (6, 6), 128: (12, 12), 512: (24, 24), 2048: (46, 46)}
    side_len_x, side_len_y = side_lengths.get(M, (side_len, side_len))

    # Generate the grid of possible points
    points = np.array([(x, y) for x in range(-side_len_x + 1, side_len_x, 2)
                       for y in range(-side_len_y + 1, side_len_y, 2)])

    # Exclude the origin (0, 0)
    points = points[~np.all(points == 0, axis=1)]

    excluded_points = []
    if exclude_points > 0:
        square_size = int(np.sqrt(exclude_points // 4))
        for i in range(square_size):
            for j in range(square_size):
                excluded_points.extend([
                    (side_len_x - 1 - 2 * i, side_len_y - 1 - 2 * j),
                    (-side_len_x + 1 + 2 * i, side_len_y - 1 - 2 * j),
                    (side_len_x - 1 - 2 * i, -side_len_y + 1 + 2 * j),
                    (-side_len_x + 1 + 2 * i, -side_len_y + 1 + 2 * j)
                ])
        excluded_points = np.array(excluded_points)

        # Filter out excluded points from the main list
        mask = np.ones(len(points), dtype=bool)
        for excluded_point in excluded_points:
            mask &= ~np.all(points == excluded_point, axis=1)
        points = points[mask]

    return points, excluded_points


def plot_qam_constellation(M):
    """
    Plot QAM constellation diagram and save to PDF along with a symbol table.

    Parameters:
        M (int): The size of the QAM constellation.
    """
    exclusions = {32: 4, 128: 16, 512: 64, 2048: 196}
    exclude_points = exclusions.get(M, 0)

    points, excluded_points = generate_qam_constellation(M, exclude_points)
    energy = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    phase = np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)
    phase_pi = phase / np.pi

    sort_indices = np.argsort(phase)
    points, energy, phase_pi = points[sort_indices], energy[sort_indices], phase_pi[sort_indices]

    # Plot constellation
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Included Symbols')
    if len(excluded_points) > 0:
        plt.scatter(excluded_points[:, 0], excluded_points[:, 1], color='red', marker='x', label='Excluded Symbols')
    plt.grid(True)
    plt.title(f'{M}-QAM Constellation Diagram')
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.gca().set_aspect('equal')
    plt.legend()
    max_val = np.max(np.abs(points)) + 2
    ticks = np.arange(-max_val, max_val + 1, 2)
    plt.xticks(ticks[ticks != 0])
    plt.yticks(ticks[ticks != 0])
    plt.axhline(0, color='black', linewidth=1.2)
    plt.axvline(0, color='black', linewidth=1.2)

    desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    pdf_filename = os.path.join(desktop_path, f'{M}-QAM_constellation.pdf')
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig()
    plt.close()

    # Create symbol table
    symbol_numbers = np.arange(1, len(points) + 1)
    chunk_size = 20
    num_chunks = math.ceil(len(points) / chunk_size)
    table_pdf_filename = os.path.join(desktop_path, f'{M}-QAM_table.pdf')

    with PdfPages(table_pdf_filename) as pdf:
        for i in range(num_chunks):
            start_idx, end_idx = i * chunk_size, min((i + 1) * chunk_size, len(points))
            fig_table = plt.figure(figsize=(8, 6))
            ax_table = fig_table.add_subplot(1, 1, 1)
            ax_table.axis('off')

            table_data = [[num, q, i, f"{e:.2f}", f"{p:.2f}π"]
                          for num, (q, i), e, p in
                          zip(symbol_numbers[start_idx:end_idx], points[start_idx:end_idx], energy[start_idx:end_idx],
                              phase_pi[start_idx:end_idx])]

            table = ax_table.table(cellText=table_data,
                                   colLabels=["Symbol", "Q", "I", "Energy", "Phase (rad)"],
                                   loc="center", cellLoc="center", colWidths=[0.2] * 5)
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)

            pdf.savefig(fig_table)
            plt.close(fig_table)


# Main execution
if __name__ == "__main__":
    M = int(input("Enter the size of QAM (4, 16, 32, 64, 128, 512, 1024, 2048, 4096): "))
    valid_sizes = [4, 16, 32, 64, 128, 512, 1024, 2048, 4096]
    if M in valid_sizes:
        print("Generating QAM constellation and table. Exporting to PDF...")
        plot_qam_constellation(M)
        print("PDF files generated successfully to your Desktop!")
    else:
        print(f"Invalid QAM size. Please choose from: {valid_sizes}.")
