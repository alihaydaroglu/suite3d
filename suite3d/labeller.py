import numpy as n
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy.ndimage
from torch.utils.data import Dataset, DataLoader
from . import plot_utils as plot
from matplotlib import pyplot as plt

class Box3D:
    def __init__(self, data):
        """
        data: Tensor or array of shape (n_roi, nz, ny, nx)
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def clone(self):
        return Box3D(self.data.clone())

# --- Box3D augmentations ---
def rotate_2d(boxes, angle=None):
    if angle is None:
        angle = n.random.uniform(0,360)
    rotated = []
    for box in boxes:
        rotated_box = scipy.ndimage.rotate(box.numpy(), angle, axes=(1,2), reshape=False, order=1)
        rotated.append(torch.tensor(rotated_box))
    return torch.stack(rotated)

def mirror_axis(boxes, axis=2):
    return torch.flip(boxes, dims=[axis])

def add_gaussian_noise(boxes, std=0.1, exclude_idx = []):
    noises = torch.zeros_like(boxes)
    for i,box in enumerate(boxes):
        if i not in exclude_idx:
            noises[i] = torch.randn_like(box) * (box.std() * std)
    return boxes + noises

def rotate_2d_torch(boxes, angle):
    """
    Rotates a batch of 3D tensors by a given angle using PyTorch.

    Args:
        boxes (torch.Tensor): A tensor of shape (nb, nz, ny, nx) representing the batch of 3D boxes.
        angle (float): The angle of rotation in degrees.

    Returns:
        torch.Tensor: A tensor of the same shape as boxes, containing the rotated boxes.
    """
    nb, nz, ny, nx = boxes.shape
    rotated_boxes = torch.zeros_like(boxes)

    #Efficiently rotate using PyTorch's affine_grid and grid_sample
    theta = torch.tensor([[torch.cos(torch.deg2rad(angle)), -torch.sin(torch.deg2rad(angle)), 0],
                          [torch.sin(torch.deg2rad(angle)), torch.cos(torch.deg2rad(angle)), 0]], dtype=torch.float32)

    grid = F.affine_grid(theta.unsqueeze(0).repeat(nb,1,1), boxes.shape)
    rotated_boxes = F.grid_sample(boxes.unsqueeze(1), grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(1)

    return rotated_boxes

def scale_boxes(boxes, factors=None, range = (0.75, 1.25)):
    if factors is None:
        factors = n.random.uniform(range[0], range[1], len(boxes))
    scaled_boxes = []
    for i in range(len(boxes)):
        scaled_boxes.append(boxes[i] * factors[i])
    return torch.stack(scaled_boxes)

class Trace1D:
    def __init__(self, data):
        """
        data: Tensor or array of shape (n_roi, n_timepoints)
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def apply_transform(self, transform):
        self.data = transform(self.data)

    def clone(self):
        return Trace1D(self.data.clone())

# --- Trace1D augmentations ---
def shuffle_snippets(traces, snippet_length = 100, permutation=None, subset = 0.5):
    n_traces, n_time = traces.shape
    n_chunks = n_time // snippet_length
    if permutation is None:
        permutation = n.random.permutation(n_chunks)
    if subset is not None:
        permutation =  permutation[:int(len(permutation) * subset)]
    shuffled = torch.zeros((n_traces, len(permutation) * snippet_length))
    for i in range((n_traces)):
        chunks = [traces[i, j:j+snippet_length] for j in range(0, n_time, snippet_length)]
        chunks_shuff = [chunks[idx] for idx in permutation]
        shuffled_trace = torch.cat(chunks_shuff, dim=0)[:n_time]  # crop in case last chunk is short
        shuffled[i] = shuffled_trace
    return shuffled



def scale_intensity(traces, factor=None, range = (0.75, 1.25)):
    if factor is None:
        factor = random.uniform(*range)
    return traces * factor

class ROIDataset(Dataset):
    def __init__(self, box3ds, trace1ds):
        """
        box3ds: List of Box3D instances
        trace1ds: List of Trace1D instances
        """
        self.box3ds = box3ds
        self.trace1ds = trace1ds
        self.n_roi = self.box3ds[0].data.shape[0] if self.box3ds else self.trace1ds[0].data.shape[0]

    def __len__(self):
        return self.n_roi

    def __getitem__(self, idx):
        box_data = [b.data[idx] for b in self.box3ds]
        trace_data = [t.data[idx] for t in self.trace1ds]
        return {
            'boxes': torch.stack(box_data),  # shape: (n_boxes, nz, ny, nx)
            'traces': torch.stack(trace_data),  # shape: (n_traces, n_timepoints)
            'roi_id': idx
        }
    # write a function that applies a box transform and trace transform to all rois, looping over each ROI, similar to how it is done in ContrsativePairDataset, and constructs a new ROIDataset of the transformed rois
    # do this by explicitly looping over all rois, and calling box_transform(roi['boxes']) and trace_transform(roi['traces']) where box_transform and trace_transform are arguments to the function
    def apply_transforms(self, box_transform, trace_transform):
        """
        Applies box_transform and trace_transform to all ROIs in the dataset.

        Args:
            box_transform (callable): Function to transform boxes.
            trace_transform (callable): Function to transform traces.

        Returns:
            ROIDataset: A new ROIDataset with transformed ROIs.
        """
        transformed_boxes = []
        transformed_traces = []
        for idx in range(len(self)):
            if idx % 100 == 0:
                print(f"Applying transforms to ROI {idx} of {len(self)}")
            roi = self[idx]
            transformed_box = roi['boxes']
            for transform in box_transform:
                transformed_box = transform(transformed_box)
            transformed_trace = roi['traces']
            for transform in trace_transform:
                transformed_trace = transform(transformed_trace)    
            transformed_boxes.append(transformed_box)
            transformed_traces.append(transformed_trace)

        # to construct a new ROIDataset, we need to de-interleave the transformed_boxes
        # the output ROIDataset should have len(roi['boxes']) Box3ds
        box3d_list = [Box3D(torch.stack([transformed_boxes[idx][i] for idx in range(len(self))])) for i in range(len(transformed_boxes[0]))]
        trace1d_list = [Trace1D(torch.stack([transformed_traces[idx][i] for idx in range(len(self))])) for i in range(len(transformed_traces[0]))]
        return ROIDataset(box3d_list, trace1d_list)
        
        



class ContrastivePairDataset(Dataset):
    def __init__(self, base_dataset, box_transforms=None, trace_transforms=None):
        """
        base_dataset: a ROIDataset instance
        box_transforms: list of transforms to apply to boxes
        trace_transforms: list of transforms to apply to traces
        """
        self.base_dataset = base_dataset
        self.box_transforms = box_transforms or {0: []} # Default to empty lists
        self.trace_transforms = trace_transforms or {0: []}
        self.box_transform_keys = list(self.box_transforms.keys())
        self.trace_transform_keys = list(self.trace_transforms.keys())
        


    def apply_transforms(self, roi_data, box_transform_idx=0, trace_transform_idx=0):
        """Applies transforms to concatenated ROI data."""
        transformed_roi = {}
        boxes = roi_data['boxes']
        traces = roi_data['traces']

        # Concatenate boxes along the batch dimension

        # print(boxes.shape)
        # Apply transforms to concatenated boxes
        transform = self.box_transforms.get(box_transform_idx, [])
        for transform_func in transform:
            boxes = transform_func(boxes)
        transformed_roi['boxes'] = boxes

        # Concatenate traces along the batch dimension

        # Apply transforms to concatenated traces
        transform = self.trace_transforms.get(trace_transform_idx, [])
        for transform_func in transform:    
            traces = transform_func(traces)
        transformed_roi['traces'] = traces

        return transformed_roi

    def __getitem__(self, idx):
        roi_data = self.base_dataset[idx]

        # Apply transforms to create two augmented views
        view1 = self.apply_transforms(roi_data.copy(), n.random.choice(self.box_transform_keys), n.random.choice(self.trace_transform_keys))
        view2 = self.apply_transforms(roi_data.copy(), n.random.choice(self.box_transform_keys), n.random.choice(self.trace_transform_keys))

        # Recombine into a dictionary for consistency
        view1['roi_id'] = idx
        view2['roi_id'] = idx
        return {'view1': view1, 'view2': view2}

    def __len__(self):
        return len(self.base_dataset)


# --- Model ---
class CellNet(nn.Module):
    def __init__(self, box_shape, trace_length, n_boxes=4, n_traces=2, latent_dim=128,
                 n_conv3d_layers=2, n_conv1d_layers=2, conv3d_kernel_sizes=None, conv1d_kernel_sizes=None,
                 conv1d_kernel_expansion = 2, conv3d_kernel_expansion = 2,
                 proj_hidden_dim=256):
        super().__init__()
        self.n_boxes = n_boxes
        self.n_traces = n_traces

        # Defaults
        if conv3d_kernel_sizes is None:
            conv3d_kernel_sizes = [3] * n_conv3d_layers
        if conv1d_kernel_sizes is None:
            conv1d_kernel_sizes = [5] * n_conv1d_layers

        # 3D Conv Encoder
        conv3d_layers = []
        in_channels = n_boxes
        for i in range(n_conv3d_layers):
            out_channels = conv3d_kernel_expansion * (2 ** i)
            conv3d_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=conv3d_kernel_sizes[i], padding=conv3d_kernel_sizes[i] // 2))
            conv3d_layers.append(nn.ReLU())
            conv3d_layers.append(nn.MaxPool3d(2))
            in_channels = out_channels
        conv3d_layers.append(nn.Flatten())
        self.box_encoder = nn.Sequential(*conv3d_layers)

        dummy_box = torch.zeros(1, n_boxes, *box_shape)
        with torch.no_grad():
            box_feat_dim = self.box_encoder(dummy_box).shape[1]
            print(self.box_encoder(dummy_box).shape)

        # 1D Conv Encoder
        conv1d_layers = []
        in_channels = n_traces
        for i in range(n_conv1d_layers):
            out_channels = conv1d_kernel_expansion * (2 ** i)
            conv1d_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=conv1d_kernel_sizes[i], 
                                           padding=conv1d_kernel_sizes[i] // 2, stride = 1))
            conv1d_layers.append(nn.ReLU())
            conv1d_layers.append(nn.MaxPool1d(2))
            in_channels = out_channels
        conv1d_layers.append(nn.AdaptiveAvgPool1d(1))
        conv1d_layers.append(nn.Flatten())
        self.trace_encoder = nn.Sequential(*conv1d_layers)

        dummy_trace = torch.zeros(1, n_traces, trace_length)
        with torch.no_grad():
            trace_feat_dim = self.trace_encoder(dummy_trace).shape[1]
            print(self.trace_encoder(dummy_trace).shape)
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(box_feat_dim + trace_feat_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, latent_dim)
        )

    def forward(self, boxes, traces):
        x_box = self.box_encoder(boxes)
        x_trace = self.trace_encoder(traces)
        x = torch.cat([x_box, x_trace], dim=1)
        return self.projection_head(x)

# --- Dataloader Utility ---
def create_dataloader(box3ds, trace1ds, batch_size=32, shuffle=True):
    dataset = ROIDataset(box3ds, trace1ds)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def nt_xent_loss(embeddings, temperature=0.5):
    """
    Computes normalized temperature-scaled cross entropy loss (NT-Xent)
    Args:
        embeddings: tensor of shape (2*B, D) with L2-normalized embeddings
    """
    batch_size = embeddings.shape[0] // 2
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    sim_matrix /= temperature

    labels = torch.arange(batch_size).repeat(2).to(embeddings.device)
    masks = torch.eye(2 * batch_size, dtype=torch.bool).to(embeddings.device)

    # mask self-similarity
    sim_matrix = sim_matrix.masked_fill(masks, -9e15)

    positives = torch.cat([torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)])
    negatives = sim_matrix[~masks].view(2 * batch_size, -1)

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    loss = F.cross_entropy(logits, torch.zeros(2 * batch_size, dtype=torch.long, device=embeddings.device))
    return loss

def train_contrastive(model, dataloader, optimizer, device, epochs=10, temperature=0.5):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0.0
        # for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        for batch_idx, batch in enumerate(dataloader):
            # if batch_idx % 50 == 0:
                # print(f"Batch {batch_idx} of {len(dataloader)}")
            boxes1 = batch['view1']['boxes'].to(device)
            traces1 = batch['view1']['traces'].to(device)
            boxes2 = batch['view2']['boxes'].to(device)
            traces2 = batch['view2']['traces'].to(device)

            # Combine views
            boxes = torch.cat([boxes1, boxes2], dim=0)
            traces = torch.cat([traces1, traces2], dim=0)

            embeddings = model(boxes, traces)
            embeddings = F.normalize(embeddings, dim=1)

            loss = nt_xent_loss(embeddings, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")


def get_embeddings(model, dataloader, device):
    """
    Generates embeddings for a dataset using a trained CellNet.

    Args:
        model: Trained CellNet model.
        dataloader: DataLoader for the unpaired dataset.
        device: Device to run the model on ('cpu' or 'cuda').

    Returns:
        A dictionary containing embeddings for boxes and traces, and roi_ids.  Returns None if an error occurs.
    """
    try:
        model.eval()
        model.to(device)
        all_embeddings = []
        all_roi_ids = []

        with torch.no_grad():
            for batch in dataloader:
                boxes = batch['boxes'].to(device)
                traces = batch['traces'].to(device)
                embeddings = model(boxes, traces)
                all_embeddings.append(embeddings.cpu())
                all_roi_ids.append(batch['roi_id'].cpu())

        return {
            'embeddings': torch.cat(all_embeddings),
            'roi_ids': torch.cat(all_roi_ids)
        }
    except Exception as e:
        print(f"An error occurred during embedding generation: {e}")
        return None

def get_cell_centroids(coords, lams, round=False):
    """
    Compute the weighted centroids of cells from Suite3D

    Args:
        coords (list): list of cell coords, output of Suite3D
        lams (list): List of pixel weights per cell, output of Suite3D

    Returns:
        centroids: (n_cells, n_dim) array of weighted centroids
    """
    n_cells = len(coords)
    ndim = len(coords[0])
    centroids = n.zeros((n_cells, ndim))

    # loop over all cells
    for cell_idx in range(n_cells):
        coord = coords[cell_idx]
        lam = lams[cell_idx]
        lam_sum = lam.sum()
        # for each coordinate, take the weighted average
        # weighted by the 'lam' values
        centroids[cell_idx] = [(coord[i] * lam).sum() / lam_sum for i in range(ndim)]

    if round:
        centroids = n.round(centroids).astype(int)

    return centroids

def fill_lam_box(meds, coords, lams, box_size):
    # meds is of size n_rois, 3 where each element is a z/y/x coordinate marking the center of a cell
    # coords is a list of n_rois elements, each element is tuple of 3 lists, 
    #     each list is of length n_pix_roi (different for each roi), listing the z/y/x coordinates of a given roi]
    #     for each roi, med will be in its coords
    # lams is a list of n_rois elements, each element is a list of n_pix_roi, corresponding a weight for each of the coordinates in coords 
    # the output should be similar to extract_padded_boxes, of size n_rois, box_size[0], box_size[1], box_size[2]
    # the center of the output box should correspond to the lam value corresponding to the med of an roi
    # remaining values of the output box should be filled with the weights of each coordinate 
    # 
    boxes = []
    for i, (cz, cy, cx) in enumerate(n.array(meds).astype(int)):
        bz, by, bx = box_size
        hz, hy, hx = bz // 2, by // 2, bx // 2

        z0, z1 = cz - hz, cz + hz + bz % 2
        y0, y1 = cy - hy, cy + hy + by % 2
        x0, x1 = cx - hx, cx + hx + bx % 2

        box = n.zeros(box_size)
        for j, (z,y,x) in enumerate(zip(coords[i][0], coords[i][1], coords[i][2])):
            if z0 <= z < z1 and y0 <= y < y1 and x0 <= x < x1:
                box[z-z0, y-y0, x-x0] = lams[i][j]
        boxes.append(box)
    return n.stack(boxes)
    


def extract_padded_boxes(mean_img, meds, box_size):
    meds = n.array(meds)
    bz, by, bx = box_size
    hz, hy, hx = bz // 2, by // 2, bx // 2

    boxes = []
    for cz, cy, cx in meds.astype(int):
        z0, z1 = cz - hz, cz + hz + bz % 2
        y0, y1 = cy - hy, cy + hy + by % 2
        x0, x1 = cx - hx, cx + hx + bx % 2

        sz0, sz1 = max(z0, 0), min(z1, mean_img.shape[0])
        sy0, sy1 = max(y0, 0), min(y1, mean_img.shape[1])
        sx0, sx1 = max(x0, 0), min(x1, mean_img.shape[2])

        cropped = mean_img[sz0:sz1, sy0:sy1, sx0:sx1]
        pad = ((max(0, -z0), max(0, z1 - mean_img.shape[0])),
               (max(0, -y0), max(0, y1 - mean_img.shape[1])),
               (max(0, -x0), max(0, x1 - mean_img.shape[2])))
        boxes.append(n.pad(cropped, pad, mode='constant'))

    return n.stack(boxes)

def plot_roi(roi, scale = 2, ntplot = 500):
    boxes = roi['boxes'].cpu().numpy()
    nb, nz, ny, nx = boxes.shape
    traces = roi['traces'].cpu().numpy()
    nf, nt = traces.shape
    nb = len(boxes)

    fy = 4; fx = nb
    f,axs = plt.subplots(fy, fx, dpi=100, height_ratios = (1.0, nz/nx, nz/nx, 2*nz/nx), layout='constrained',sharex=True)

    for i in range(nb):
        plot.show_img(boxes[i][nz//2], cmap='Greys', ax = axs[0][i])
        plot.show_img(boxes[i][:,ny//2], cmap='Greys', ax = axs[1][i])
        plot.show_img(boxes[i][:,:,nx//2], cmap='Greys', ax = axs[2][i])

    gs = axs[3,0].get_gridspec()
    for ax in axs[3, :]:
        ax.remove()
    ax = f.add_subplot(gs[3, :])
    plot_idx = n.argmax(traces[0])
    for i in range(nf):
        ax.plot(traces[i][max(plot_idx-ntplot, 0):plot_idx+ntplot])
    ax.set_yticks([])
    ax.set_xticks([])