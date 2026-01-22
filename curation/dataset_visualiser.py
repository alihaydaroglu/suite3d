"""
Comprehensive visualisation and sanity checking tool for large suite3d datasets.
Handles datasets with hundreds of thousands of cells efficiently.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional, Dict


class Suite3DVisualiser:
    """
    Memory-efficient visualisation tool for large suite3d datasets.
    """
    
    def __init__(self, patches_file: str, info_file: str, use_memmap: bool = True):
        """
        Initialize visualiser.
        
        Args:
            patches_file: Path to all_sessions_patches.npy
            info_file: Path to all_sessions_info.npy  
            use_memmap: Use memory mapping to avoid loading full array
        """
        self.patches_file = Path(patches_file)
        self.info_file = Path(info_file)
        self.use_memmap = use_memmap
        
        if not self.patches_file.exists():
            raise FileNotFoundError(f"Patches file not found: {patches_file}")
        if not self.info_file.exists():
            raise FileNotFoundError(f"Info file not found: {info_file}")
        
        # Load info
        print("Loading dataset info...")
        self.info = np.load(info_file, allow_pickle=True).item()
        
        # Load patches (with memory mapping if requested)
        print("Loading patches dataset...")
        if use_memmap:
            print("Using memory mapping (data stays on disk)...")
            self.patches = np.load(patches_file, mmap_mode='r')
        else:
            print("Loading full dataset into memory (may take time)...")
            self.patches = np.load(patches_file)
        
        self.n_cells, self.n_channels, self.nz, self.ny, self.nx = self.patches.shape
        
        print(f"Dataset loaded!")
        print(f"  Shape: {self.patches.shape}")
        print(f"  Memory mapping: {use_memmap}")
        print(f"  Data type: {self.patches.dtype}")
        print(f"  File size: {self.patches_file.stat().st_size / 1e9:.2f} GB")
        
        # Channel names for clarity
        self.channel_names = ['Mean Image', 'Correlation Map', 'Cell Footprint']
        
    def get_basic_stats(self) -> Dict:
        """Get basic dataset statistics (memory efficient)."""
        print("\nComputing dataset statistics...")
        
        # Sample for statistics to avoid memory issues
        sample_size = min(10000, self.n_cells)
        sample_indices = np.random.choice(self.n_cells, sample_size, replace=False)
        sample_data = self.patches[sample_indices]
        
        stats = {
            'n_cells': self.n_cells,
            'shape': self.patches.shape,
            'dtype': str(self.patches.dtype),
            'sample_size_for_stats': sample_size
        }
        
        for ch in range(self.n_channels):
            ch_data = sample_data[:, ch]
            stats[f'ch{ch}_mean'] = np.mean(ch_data)
            stats[f'ch{ch}_std'] = np.std(ch_data)
            stats[f'ch{ch}_min'] = np.min(ch_data)
            stats[f'ch{ch}_max'] = np.max(ch_data)
            stats[f'ch{ch}_nonzero_frac'] = np.mean(ch_data > 0)
        
        return stats
    
    def print_dataset_summary(self):
        """Print comprehensive dataset summary."""
        stats = self.get_basic_stats()
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        print(f"Total cells: {stats['n_cells']:,}")
        print(f"Shape: {stats['shape']}")
        print(f"Data type: {stats['dtype']}")
        print(f"Channels: {self.n_channels} ({', '.join(self.channel_names)})")
        print(f"Patch size: {self.nz} × {self.ny} × {self.nx} (Z × Y × X)")
        
        if 'session_names' in self.info:
            print(f"Sessions: {len(self.info['session_names'])}")
            print(f"Session names: {', '.join(self.info['session_names'][:5])}")
            if len(self.info['session_names']) > 5:
                print(f"  ... and {len(self.info['session_names'])-5} more")
        
        print(f"\nStatistics (based on {stats['sample_size_for_stats']:,} random cells):")
        print("-" * 40)
        
        for ch in range(self.n_channels):
            ch_name = self.channel_names[ch]
            print(f"{ch_name} (Channel {ch}):")
            print(f"  Mean: {stats[f'ch{ch}_mean']:.4f}")
            print(f"  Std:  {stats[f'ch{ch}_std']:.4f}")
            print(f"  Range: [{stats[f'ch{ch}_min']:.4f}, {stats[f'ch{ch}_max']:.4f}]")
            print(f"  Non-zero: {stats[f'ch{ch}_nonzero_frac']*100:.1f}%")
        
        print("="*60)
    
    def visualise_random_samples(self, n_samples: int = 16, figsize: Tuple[int, int] = (20, 12), 
                                save_path: Optional[str] = None):
        """
        Visualise random cell samples from all channels.
        
        Args:
            n_samples: Number of cells to visualise
            figsize: Figure size
            save_path: Optional path to save figure
        """
        # Select random cells
        cell_indices = np.random.choice(self.n_cells, n_samples, replace=False)
        
        # Create grid: n_samples columns, 3 channels + 2 more views of 3D image
        fig, axes = plt.subplots(5, n_samples, figsize=figsize)
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        
        print(f"\nVisualising {n_samples} random cells (indices: {cell_indices[:10]}{'...' if n_samples > 10 else ''})")
        
        for i, cell_idx in enumerate(cell_indices):
            # Load single cell data
            cell_data = self.patches[cell_idx]  # Shape: (3, 5, 20, 20)
            
            # Show each channel (max projection for 3D)
            for ch in range(3):
                channel_data = cell_data[ch]  # Shape: (5, 20, 20)
                
                # Max projection along Z-axis
                max_proj = np.max(channel_data, axis=0)  # Shape: (20, 20)
                
                # Choose colormap based on channel
                if ch == 0:  # Mean image
                    cmap = 'gray'
                    vmin, vmax = None, None
                elif ch == 1:  # Correlation map
                    cmap = 'viridis' 
                    vmin, vmax = None, None
                else:  # Cell footprint
                    cmap = 'hot'
                    vmin = 0
                    vmax = np.max(max_proj) if np.max(max_proj) > 0 else 1
                
                im = axes[ch, i].imshow(max_proj, cmap=cmap, vmin=vmin, vmax=vmax)
                
                if i == 0:  # Add channel labels on first column
                    axes[ch, i].set_ylabel(f'{self.channel_names[ch]}\n(Max Proj)', fontsize=10)
                
                axes[ch, i].set_title(f'Cell {cell_idx}' if ch == 0 else '', fontsize=8)
                axes[ch, i].axis('off')
            
            # for composite view
            # # Create composite view (all channels overlaid or side by side)
            # mean_proj = np.max(cell_data[0], axis=0)
            # footprint_proj = np.max(cell_data[2], axis=0)
            
            # # Normalize for overlay
            # if np.max(mean_proj) > 0:
            #     mean_norm = mean_proj / np.max(mean_proj)
            # else:
            #     mean_norm = mean_proj
                
            # if np.max(footprint_proj) > 0:
            #     footprint_norm = footprint_proj / np.max(footprint_proj)
            # else:
            #     footprint_norm = footprint_proj
            
            # # Create RGB composite: mean in gray, footprint in red
            # composite = np.zeros((20, 20, 3))
            # composite[:, :, 0] = footprint_norm  # Red channel = cell footprint
            # composite[:, :, 1] = mean_norm       # Green channel = mean image
            # composite[:, :, 2] = mean_norm       # Blue channel = mean image
            
            # axes[3, i].imshow(composite)
            # if i == 0:
            #     axes[3, i].set_ylabel('Composite\n(Gray=Mean, Red=Footprint)', fontsize=10)
            # axes[3, i].axis('off')

            channel_data = cell_data[0]             # mean image, (5, 20, 20)
            middle_z_slice = channel_data[self.nz // 2]  # Middle Z slice
            middle_y_slice = channel_data[:, self.ny // 2]  # Middle Y slice

            axes[3, i].imshow(middle_z_slice, cmap='gray')
            if i == 0:
                axes[3, i].set_ylabel('Middle Z Slice\n(Gray=Mean, Red=Footprint)', fontsize=10)
            axes[3, i].axis('off')
            axes[4, i].imshow(middle_y_slice, cmap='gray')
            if i == 0:
                axes[4, i].set_ylabel('Middle Y Slice\n(Gray=Mean, Red=Footprint)', fontsize=10)
            axes[4, i].axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def analyze_channel_distributions(self, sample_size: int = 50000, 
                                    save_path: Optional[str] = None):
        """
        Analyze and plot intensity distributions for all channels.
        
        Args:
            sample_size: Number of cells to sample for statistics
            save_path: Optional path to save figure
        """
        print(f"\nAnalyzing intensity distributions using {sample_size:,} random cells...")
        
        # Sample data to avoid memory issues
        sample_indices = np.random.choice(self.n_cells, 
                                        min(sample_size, self.n_cells), 
                                        replace=False)
        sample_data = self.patches[sample_indices]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Per-pixel intensity distributions
        for ch in range(3):
            ch_data = sample_data[:, ch].flatten()
            
            axes[0, ch].hist(ch_data, bins=100, alpha=0.7, color=f'C{ch}', density=True)
            axes[0, ch].set_title(f'{self.channel_names[ch]}\nPer-Pixel Intensities')
            axes[0, ch].set_xlabel('Intensity')
            axes[0, ch].set_ylabel('Density')
            axes[0, ch].set_yscale('log')
            
            # Add statistics text
            stats_text = f'Mean: {np.mean(ch_data):.3f}\n'
            stats_text += f'Std: {np.std(ch_data):.3f}\n'
            stats_text += f'Non-zero: {np.mean(ch_data > 0)*100:.1f}%'
            axes[0, ch].text(0.02, 0.98, stats_text, transform=axes[0, ch].transAxes,
                           verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Per-cell mean intensities
        cell_means = np.mean(sample_data, axis=(2, 3, 4))  # Average over spatial dimensions
        
        for ch in range(3):
            axes[1, ch].hist(cell_means[:, ch], bins=50, alpha=0.7, color=f'C{ch}', density=True)
            axes[1, ch].set_title(f'{self.channel_names[ch]}\nPer-Cell Mean Intensities')
            axes[1, ch].set_xlabel('Mean Intensity')
            axes[1, ch].set_ylabel('Density')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Distribution analysis saved to {save_path}")
        
        plt.show()
        
        # Print correlation analysis
        print("\nChannel correlation analysis:")
        corr_matrix = np.corrcoef(cell_means.T)
        for i in range(3):
            for j in range(i+1, 3):
                corr = corr_matrix[i, j]
                print(f"  {self.channel_names[i]} vs {self.channel_names[j]}: r = {corr:.3f}")
    
    def inspect_specific_cells(self, cell_indices: List[int], figsize: Tuple[int, int] = (15, 10),
                              save_path: Optional[str] = None):
        """
        Detailed inspection of specific cells with all z-slices.
        
        Args:
            cell_indices: List of cell indices to inspect
            figsize: Figure size
            save_path: Optional path to save figure
        """
        n_cells = len(cell_indices)
        
        # Create subplots: n_cells rows, 3 channels * nz slices columns
        fig, axes = plt.subplots(n_cells, self.n_channels * self.nz, 
                                figsize=(self.n_channels * self.nz * 2, n_cells * 3))
        
        if n_cells == 1:
            axes = axes.reshape(1, -1)
        
        for cell_i, cell_idx in enumerate(cell_indices):
            if cell_idx >= self.n_cells:
                print(f"Warning: Cell index {cell_idx} out of range (max: {self.n_cells-1})")
                continue
                
            print(f"Loading cell {cell_idx}...")
            cell_data = self.patches[cell_idx]
            
            col = 0
            for ch in range(self.n_channels):
                for z in range(self.nz):
                    slice_data = cell_data[ch, z]
                    
                    # Choose colormap
                    if ch == 0:
                        cmap = 'gray'
                    elif ch == 1:
                        cmap = 'viridis'
                    else:
                        cmap = 'hot'
                    
                    axes[cell_i, col].imshow(slice_data, cmap=cmap)
                    
                    # Add titles
                    if cell_i == 0:
                        axes[cell_i, col].set_title(f'{self.channel_names[ch][:4]}\nZ={z}', 
                                                   fontsize=8)
                    
                    axes[cell_i, col].axis('off')
                    col += 1
            
            # Add cell label
            axes[cell_i, 0].text(-0.5, 0.5, f'Cell\n{cell_idx}', 
                                transform=axes[cell_i, 0].transAxes,
                                verticalalignment='center', horizontalalignment='right',
                                fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Detailed inspection saved to {save_path}")
        
        plt.show()
    
    def validate_footprint_channel(self, n_samples: int = 20):
        """
        Validate that the cell footprint channel makes sense.
        
        Args:
            n_samples: Number of cells to validate
        """
        print(f"\nValidating cell footprint channel using {n_samples} random cells...")
        
        sample_indices = np.random.choice(self.n_cells, n_samples, replace=False)
        
        validation_results = {
            'cells_with_footprint': 0,
            'cells_without_footprint': 0,
            'footprint_sizes': [],
            'footprint_weights_range': [],
            'footprint_center_weights': []
        }
        
        for cell_idx in sample_indices:
            footprint = self.patches[cell_idx, 2]  # Channel 2 = footprint
            
            nonzero_pixels = np.sum(footprint > 0)
            
            if nonzero_pixels > 0:
                validation_results['cells_with_footprint'] += 1
                validation_results['footprint_sizes'].append(nonzero_pixels)
                
                nonzero_weights = footprint[footprint > 0]
                validation_results['footprint_weights_range'].append(
                    (np.min(nonzero_weights), np.max(nonzero_weights))
                )
                
                # Check center pixel weight (should often be high)
                center_z, center_y, center_x = self.nz//2, self.ny//2, self.nx//2
                center_weight = footprint[center_z, center_y, center_x]
                validation_results['footprint_center_weights'].append(center_weight)
            else:
                validation_results['cells_without_footprint'] += 1
        
        print(f"Footprint validation results:")
        print(f"  Cells with footprints: {validation_results['cells_with_footprint']}/{n_samples}")
        print(f"  Cells without footprints: {validation_results['cells_without_footprint']}/{n_samples}")
        
        if validation_results['footprint_sizes']:
            sizes = validation_results['footprint_sizes']
            print(f"  Footprint sizes: {np.min(sizes)} - {np.max(sizes)} pixels (mean: {np.mean(sizes):.1f})")
            
            weights_min = [r[0] for r in validation_results['footprint_weights_range']]
            weights_max = [r[1] for r in validation_results['footprint_weights_range']]
            print(f"  Weight ranges: [{np.min(weights_min):.3f}, {np.max(weights_max):.3f}]")
            
            center_weights = validation_results['footprint_center_weights']
            center_nonzero = [w for w in center_weights if w > 0]
            if center_nonzero:
                print(f"  Center pixel weights: {len(center_nonzero)}/{len(center_weights)} have weight > 0")
                print(f"    Mean center weight: {np.mean(center_nonzero):.3f}")
    
    def interactive_cell_browser(self):
        """
        Launch interactive cell browser (works best in Jupyter notebooks).
        """
        print("\nLaunching interactive cell browser...")
        print("Use slider to browse through cells, or enter specific cell index.")
        
        # Create interactive plot
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        def update_cell(cell_idx):
            cell_idx = int(cell_idx)
            if cell_idx >= self.n_cells:
                cell_idx = self.n_cells - 1
            elif cell_idx < 0:
                cell_idx = 0
                
            cell_data = self.patches[cell_idx]
            
            # Clear axes
            for ax in axes:
                ax.clear()
            
            # Show each channel
            for ch in range(3):
                max_proj = np.max(cell_data[ch], axis=0)
                
                if ch == 0:
                    cmap = 'gray'
                elif ch == 1:
                    cmap = 'viridis'
                else:
                    cmap = 'hot'
                
                axes[ch].imshow(max_proj, cmap=cmap)
                axes[ch].set_title(f'{self.channel_names[ch]}')
                axes[ch].axis('off')
            
            # Composite
            mean_proj = np.max(cell_data[0], axis=0)
            footprint_proj = np.max(cell_data[2], axis=0)
            
            if np.max(mean_proj) > 0:
                mean_norm = mean_proj / np.max(mean_proj)
            else:
                mean_norm = mean_proj
                
            composite = np.zeros((20, 20, 3))
            composite[:, :, 1] = mean_norm
            composite[:, :, 2] = mean_norm
            if np.max(footprint_proj) > 0:
                composite[:, :, 0] = footprint_proj / np.max(footprint_proj)
            
            axes[3].imshow(composite)
            axes[3].set_title('Composite')
            axes[3].axis('off')
            
            fig.suptitle(f'Cell {cell_idx} / {self.n_cells-1}')
            plt.draw()
        
        # Show first cell
        update_cell(0)
        
        # Add slider (if in interactive environment)
        try:
            from matplotlib.widgets import Slider
            ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03])
            slider = Slider(ax_slider, 'Cell Index', 0, self.n_cells-1, 
                          valinit=0, valfmt='%d')
            slider.on_changed(update_cell)
        except:
            print("Interactive slider not available in this environment")
        
        plt.tight_layout()
        plt.show()
        
        return update_cell  # Return function for manual use
    
    def find_interesting_cells(self, criteria: str = 'bright', n_cells: int = 10) -> List[int]:
        """
        Find cells that meet certain criteria.
        
        Args:
            criteria: 'bright', 'dim', 'large_footprint', 'small_footprint', 'high_corr'
            n_cells: Number of cells to return
            
        Returns:
            List of interesting cell indices
        """
        print(f"\nFinding {n_cells} cells with criteria: {criteria}")
        
        # Sample subset for efficiency
        sample_size = min(50000, self.n_cells)
        sample_indices = np.random.choice(self.n_cells, sample_size, replace=False)
        sample_data = self.patches[sample_indices]
        
        if criteria == 'bright':
            # Highest mean intensity in mean image channel
            scores = np.mean(sample_data[:, 0], axis=(1, 2, 3))
            selected = np.argsort(scores)[-n_cells:][::-1]
            
        elif criteria == 'dim':
            # Lowest mean intensity in mean image channel  
            scores = np.mean(sample_data[:, 0], axis=(1, 2, 3))
            selected = np.argsort(scores)[:n_cells]
            
        elif criteria == 'large_footprint':
            # Most pixels in footprint
            scores = np.sum(sample_data[:, 2] > 0, axis=(1, 2, 3))
            selected = np.argsort(scores)[-n_cells:][::-1]
            
        elif criteria == 'small_footprint':
            # Fewest pixels in footprint (but > 0)
            footprint_sizes = np.sum(sample_data[:, 2] > 0, axis=(1, 2, 3))
            valid_footprints = footprint_sizes > 0
            valid_indices = np.where(valid_footprints)[0]
            
            if len(valid_indices) >= n_cells:
                scores = footprint_sizes[valid_indices]
                selected_valid = np.argsort(scores)[:n_cells]
                selected = valid_indices[selected_valid]
            else:
                selected = valid_indices
                
        elif criteria == 'high_corr':
            # Highest max intensity in correlation channel
            scores = np.max(sample_data[:, 1], axis=(1, 2, 3))
            selected = np.argsort(scores)[-n_cells:][::-1]
            
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
        
        # Convert back to original indices
        interesting_indices = sample_indices[selected].tolist()
        
        print(f"Found {len(interesting_indices)} interesting cells:")
        for i, idx in enumerate(interesting_indices[:5]):  # Show first 5
            print(f"  {i+1}. Cell {idx}")
        if len(interesting_indices) > 5:
            print(f"  ... and {len(interesting_indices)-5} more")
        
        return interesting_indices
    
    def visualise_cluster_examples(self, cluster_labels_file: str, n_examples: int = 5, 
                                  figsize: Tuple[int, int] = (15, 12), save_path: Optional[str] = None):
        """
        Visualize typical examples from each cluster.
        
        Args:
            cluster_labels_file: Path to cluster labels numpy file
            n_examples: Number of examples to show per cluster (default: 5)
            figsize: Figure size (default: (15, 12))
            save_path: Optional path to save figure
        """
        print(f"\nLoading cluster labels from {cluster_labels_file}")
        
        # Load cluster labels
        cluster_labels_path = Path(cluster_labels_file)
        if not cluster_labels_path.exists():
            raise FileNotFoundError(f"Cluster labels file not found: {cluster_labels_file}")
        
        cluster_labels = np.load(cluster_labels_file)
        
        if len(cluster_labels) != self.n_cells:
            raise ValueError(f"Cluster labels length ({len(cluster_labels)}) doesn't match dataset size ({self.n_cells})")
        
        # Get unique clusters, ignore -1 (noise)
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = [c for c in unique_clusters if c != -1]
        n_clusters = len(unique_clusters)
        print(f"Found {n_clusters} unique clusters (excluding noise): {unique_clusters}")
        
        # For each cluster, find typical examples (closest to cluster centroid)
        print("Computing cluster centroids and finding representative examples...")
        
        cluster_examples = {}
        for cluster_id in unique_clusters:
            # Get all cells in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            n_cells_in_cluster = len(cluster_indices)
            
            print(f"  Cluster {cluster_id}: {n_cells_in_cluster} cells")
            
            if n_cells_in_cluster == 0:
                continue
                
            # Sample a subset if cluster is too large (for efficiency)
            if n_cells_in_cluster > 1000:
                sample_indices = np.random.choice(cluster_indices, 1000, replace=False)
            else:
                sample_indices = cluster_indices
            
            # Load data for sampled cells
            cluster_data = self.patches[sample_indices]
            
            # Compute centroid (mean across all cells in cluster)
            # Shape: (n_channels, nz, ny, nx)
            centroid = np.mean(cluster_data, axis=0)
            
            # Find cells closest to centroid
            distances = []
            for i, cell_idx in enumerate(sample_indices):
                cell_data = cluster_data[i]
                # Compute Euclidean distance to centroid
                distance = np.linalg.norm(cell_data - centroid)
                distances.append((distance, cell_idx))
            
            # Sort by distance and take closest examples
            distances.sort(key=lambda x: x[0])
            n_examples_to_show = min(n_examples, len(distances))
            representative_indices = [distances[i][1] for i in range(n_examples_to_show)]
            cluster_examples[cluster_id] = representative_indices


        # Create visualization
        # Layout: n_clusters rows, n_examples*3 columns (3 images per sample)
        n_cols = n_examples * 3
        fig, axes = plt.subplots(n_clusters, n_cols, figsize=(n_cols*1.2, n_clusters*1.2), squeeze=False)

        print("\nGenerating visualization...")

        for cluster_row, cluster_id in enumerate(unique_clusters):
            if cluster_id not in cluster_examples:
                continue
            examples = cluster_examples[cluster_id]
            for example_idx, cell_idx in enumerate(examples):
                cell_data = self.patches[cell_idx]
                for ch in range(3):
                    col = example_idx * 3 + ch
                    if ch == 0:
                        img = np.max(cell_data[0], axis=0)
                        cmap = 'gray'
                        title = 'Mean'
                    elif ch == 1:
                        img = np.max(cell_data[1], axis=0)
                        cmap = 'viridis'
                        title = 'Corr'
                    else:
                        img = np.max(cell_data[2], axis=0)
                        cmap = 'hot'
                        title = 'Foot'
                    axes[cluster_row, col].imshow(img, cmap=cmap)
                    axes[cluster_row, col].axis('off')
                    if cluster_row == 0:
                        axes[cluster_row, col].set_title(f'Ex{example_idx+1}\n{title}', fontsize=7)
            # Add cluster label on the left
            axes[cluster_row, 0].text(-0.5, 0.5, f'Cluster {cluster_id}\n({len(examples)} shown)',
                                      transform=axes[cluster_row, 0].transAxes,
                                      verticalalignment='center', horizontalalignment='right',
                                      fontsize=8, fontweight='bold',
                                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            # Hide empty subplots if fewer examples than requested
            for example_idx in range(len(examples), n_examples):
                for ch in range(3):
                    col = example_idx * 3 + ch
                    axes[cluster_row, col].axis('off')

        plt.subplots_adjust(wspace=0.02, hspace=0.08, left=0.03, right=0.99, top=0.95, bottom=0.03)
        # plt.suptitle(f'Cluster Examples: {n_examples} representative cells per cluster\n'
        #              f'Each sample: Mean | Corr | Foot',
        #              fontsize=12, y=0.99)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Cluster visualization saved to {save_path}")

        plt.show()

        # Print cluster statistics
        print("\nCluster Statistics:")
        print("-" * 40)
        for cluster_id in unique_clusters:
            n_cells_in_cluster = np.sum(cluster_labels == cluster_id)
            percentage = (n_cells_in_cluster / self.n_cells) * 100
            print(f"Cluster {cluster_id:2d}: {n_cells_in_cluster:6,} cells ({percentage:5.1f}%)")
    
    def export_summary_report(self, output_file: str):
        """
        Export comprehensive summary report.
        
        Args:
            output_file: Path to output text file
        """
        print(f"\nExporting summary report to {output_file}")
        
        stats = self.get_basic_stats()
        
        with open(output_file, 'w') as f:
            f.write("Suite3D Dataset Visualisation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {self.patches_file}\n")
            f.write(f"Info: {self.info_file}\n\n")
            
            f.write("Dataset Overview:\n")
            f.write(f"  Shape: {stats['shape']}\n")
            f.write(f"  Total cells: {stats['n_cells']:,}\n")
            f.write(f"  Data type: {stats['dtype']}\n")
            f.write(f"  File size: {self.patches_file.stat().st_size / 1e9:.2f} GB\n")
            f.write(f"  Memory mapping used: {self.use_memmap}\n\n")
            
            f.write("Channels:\n")
            for i, name in enumerate(self.channel_names):
                f.write(f"  {i}: {name}\n")
            f.write("\n")
            
            f.write(f"Statistics (sample size: {stats['sample_size_for_stats']:,}):\n")
            for ch in range(self.n_channels):
                f.write(f"  {self.channel_names[ch]}:\n")
                f.write(f"    Mean: {stats[f'ch{ch}_mean']:.4f}\n")
                f.write(f"    Std: {stats[f'ch{ch}_std']:.4f}\n")
                f.write(f"    Range: [{stats[f'ch{ch}_min']:.4f}, {stats[f'ch{ch}_max']:.4f}]\n")
                f.write(f"    Non-zero fraction: {stats[f'ch{ch}_nonzero_frac']*100:.1f}%\n\n")
            
            if hasattr(self, 'info'):
                f.write("Session Info:\n")
                for key, value in self.info.items():
                    f.write(f"  {key}: {value}\n")
        
        print(f"Report saved to {output_file}")

    def visualise_cluster_arrays_side_by_side(self, arr1: np.ndarray, arr2: np.ndarray, figsize: Tuple[int, int] = (10, 2), save_path: Optional[str] = None):
        """
        Visualise two arrays of shape (N, 3, 5, 20, 20) side by side.
        Each row is a cluster, columns are arr1 and arr2.
        Args:
            arr1: np.ndarray of shape (N, 3, 5, 20, 20)
            arr2: np.ndarray of shape (N, 3, 5, 20, 20)
            figsize: Figure size per row (default: (10, 2))
            save_path: Optional path to save figure
        """
        assert arr1.shape == arr2.shape, "Arrays must have the same shape."
        assert arr1.ndim == 5 and arr1.shape[1] == 3, "Arrays must be of shape (N, 3, 5, 20, 20)."
        N = arr1.shape[0]
        row_labels = ['Mean', 'Median']
        col_labels = ['Noise'] + [f'Cluster {i}' for i in range(1, N)]
        # Make each subplot smaller and tighten layout
        fig, axes = plt.subplots(2, N, figsize=(1.8*N, 2.8), squeeze=False)
        arrays = [arr1, arr2]
        for row in range(2):
            for col in range(N):
                img = np.max(arrays[row][col, 0], axis=0)
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].axis('off')
                if row == 0:
                    axes[row, col].set_title(col_labels[col], fontsize=8)
                if col == 0:
                    axes[row, col].set_ylabel(row_labels[row], fontsize=10, fontweight='bold')
        plt.subplots_adjust(wspace=0.01, hspace=0.05, left=0.03, right=0.99, top=0.90, bottom=0.10)
        plt.suptitle('Cluster Representatives: Mean (top) and Median (bottom)', fontsize=12, y=0.97)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Cluster arrays side-by-side visualization saved to {save_path}")
        plt.show()

def main():
    """Command line interface for dataset visualisation."""

    # Initialize (uses memory mapping)
    viz = Suite3DVisualiser(r"\\path\to\all_sessions_patches.npy", r"\\path\to\all_sessions_info.npy")

    # # Quick overview
    # viz.print_dataset_summary()

    # # Visualize random samples  
    viz.visualise_random_samples(n_samples=20)



if __name__ == "__main__":
    main()