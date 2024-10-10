"""
A set of utilities for handling spatial data.
"""
from typing import List, Optional

import anndata
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import scanpy as sc
import squidpy as sq
from scipy import ndimage as ndi
import skimage.segmentation as seg
from skimage import segmentation, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import filters
from shapely import Polygon, Point
from tqdm import tqdm


def get_spatial_nn(cell: str, adata: anndata.AnnData):
    """Get neighbors of cell.

    Args:
        cell: Cell barcode
        adata: Anndata object with spatial connectivities.
    """

    cell_ind = np.where(adata.obs_names == cell)[0][0]

    spatial_connectivities = adata.obsp["spatial_connectivities"]
    spatial_distances = adata.obsp["spatial_distances"]

    neighbors = np.where(
        adata.obsp["spatial_connectivities"][cell_ind, :].todense() > 0
    )[1]
    distances = adata.obsp["spatial_distances"][cell_ind, neighbors].todense()

    return adata.obs_names[neighbors], np.ravel(distances.flatten())


def get_spatial_neighborhood_graph(
    cell: str, adata: anndata.AnnData, number_of_hops: int = 1
) -> nx.DiGraph:
    """Renders a spatial neighborhood as a graph.

    Args:
        cell: Cell barcode
        adata: Anndata with spatial coJeebie2ordinates
        number_of_hops: Number of hops to include

    Returns:
        A Networkx graph of the neighborhood.
    """

    neighborhood_graph = nx.Graph()

    hop_queue = [cell]
    neighborhood_graph.add_node(cell)
    for hops in range(number_of_hops):
        new_queue = []
        while len(hop_queue) > 0:
            node = hop_queue.pop(0)
            neighbors, distances = get_spatial_nn(node, adata)

            for neighbor, distance in zip(neighbors, distances):
                neighborhood_graph.add_edge(node, neighbor, distance=distance)
                new_queue.append(neighbor)

        hop_queue = new_queue

    return neighborhood_graph

def quantify_neighborhood(
    cell: str,
    adata: anndata.AnnData,
    variables: List[str],
    neighborhood_graph: Optional[nx.Graph] = None,
    number_of_hops: int = 1,
    normalize: bool = True,
) -> pd.DataFrame:
    """Quantifies neighborhood composition of cell.

    Args:
        cell: Cell barcode
        adata: Anndata object.
        variables: Meta data variables, must be a column in adata.obs.
        number_of_hops: Radius of neighborhood in hops.

    Returns:
        Composition of each variable in the neigbhorhood.
    """

    if neighborhood_graph is None:
        if adata is None:
            raise Exception("Must pass in adata if neighborhood graph not specified.")
        neighborhood_graph = get_spatial_neighborhood_graph(
            cell, adata, number_of_hops
        )

    composition = np.zeros((len(variables))).astype(float)
    size_of_neighborhood = 0
    # size_of_neighborhood = len(neighborhood_graph.nodes)
    for _, node in nx.bfs_edges(neighborhood_graph, cell, depth_limit=number_of_hops):
        node_data = adata.obs.loc[node, variables]
        composition += node_data.values.astype(float)
        size_of_neighborhood += 1

    if normalize:
        composition = composition / size_of_neighborhood

    return pd.DataFrame([composition], columns=variables)


def segment_tumors(
    adata: anndata.AnnData,
    annotation: str = "tumor",
    bin_size: int = 50,
    gaussian_sigma=1.5,
    clear: bool = True,
    erosion: bool = False,
    threshold_method = "otsu",
    min_distance = 0,
    try_all_filters = False,
    num_tumors = np.inf,
    verbose=True,
    copy=True,
) -> anndata.AnnData:
    """Segments spatial data using target site information.

    Adapts an imaging pipeline for segementing cell nuclei to detect regions
    that are enriched for tumor cells, which are in turn annotated as
    distinct tumors.

    Args:
        adata: Anndata with spatial coordinates. Assumes "PercentUncut" is an
            item in the metadata
        annotation: Meta data item used for detecting spots that are tumor
            (default = 'tumor')
        bin_size: Number of consecutive spots to pool together while detecting
            tumors. Decreasing this number produces higher resolution, but
            perhaps over-segmented tumors.
        verbose: Report helpful statistics.
        copy: Return a new dataframe.

    Returns:
        A new Anndata with new meta data corresponding to tumor boundaries and
            tumor identifiers.
    """

    if copy:
        st_adata = adata.copy()
    else:
        st_adata = adata

    spatial_coords = st_adata.obsm["spatial"]
    is_tumor = (
        st_adata.obs[annotation]
        .astype(bool)
        .astype(int)
        .values.reshape(len(spatial_coords), 1)
    )
    coords_to_counts = np.append(spatial_coords, is_tumor, 1)

    # create blank image
    x_bins = np.arange(0, np.max(coords_to_counts[:, 0]), step=bin_size).astype(
        int
    )
    y_bins = np.arange(0, np.max(coords_to_counts[:, 1]), step=bin_size).astype(
        int
    )

    count_image = np.zeros((len(x_bins), len(y_bins)))

    # bin counts
    x_inds = np.digitize(coords_to_counts[:, 0], x_bins)
    y_inds = np.digitize(coords_to_counts[:, 1], y_bins)

    # assign counts
    for count_index, x, y in zip(range(len(x_inds)), x_inds, y_inds):
        val = coords_to_counts[count_index, 2]
        count_image[x - 1, y - 1] += val

    count_image = np.rot90(count_image, k=3)
    count_image = np.fliplr(count_image)

    # smooth count image
    smooth = filters.gaussian(count_image, sigma=gaussian_sigma)

    if try_all_filters:
        fig, ax = filters.try_all_threshold(smooth, figsize=(10, 8), verbose=False)
        plt.show()

    edges = filters.scharr(smooth)
    smooth = ndi.median_filter(smooth, size=2)

    # find threshold for filtering
    if threshold_method == 'yen':
        thresh_value = filters.threshold_yen(smooth)
    elif threshold_method == 'otsu':
        thresh_value = filters.threshold_otsu(smooth)
    elif threshold_method == 'li':
        thresh_value = filters.threshold_li(smooth)

    thresh = smooth > thresh_value

    # identify non-overlapping tumors
    fill = ndi.binary_fill_holes(thresh)
    if clear:
        clear = fill
    else:
        clear = segmentation.clear_border(fill)
    dilate = morphology.binary_dilation(clear)
    erode = morphology.binary_erosion(clear)
    mask = np.logical_and(dilate, ~erode)

    if verbose:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6), sharey=True)

        ax[0, 0].imshow(count_image, cmap=plt.cm.gray)
        ax[0, 0].set_title("a) Raw")

        ax[0, 1].imshow(smooth, cmap=plt.cm.gray)
        ax[0, 1].set_title("b) Blur")

        ax[0, 2].imshow(thresh, cmap=plt.cm.gray)
        ax[0, 2].set_title("c) Threshold")

        ax[0, 3].imshow(fill, cmap=plt.cm.gray)
        ax[0, 3].set_title("c-1) Fill in")

        ax[1, 0].imshow(clear, cmap=plt.cm.gray)
        ax[1, 0].set_title("c-2) Keep one tumor")

        ax[1, 1].imshow(dilate, cmap=plt.cm.gray)
        ax[1, 1].set_title("d) Dilate")

        ax[1, 2].imshow(erode, cmap=plt.cm.gray)
        ax[1, 2].set_title("e) Erode")

        ax[1, 3].imshow(mask, cmap=plt.cm.gray)
        ax[1, 3].set_title("f) Tumor boundary")

        for a in ax.ravel():
            a.set_axis_off()

        fig.tight_layout()
        plt.show()

    # get labels of distinct tumors


    if erosion:
        final_mask = erode
    else:
        final_mask = dilate

    distance = ndi.distance_transform_edt(final_mask)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)),
                            num_peaks = num_tumors, labels=final_mask,
                            min_distance=min_distance)
    watershed_mask = np.zeros(distance.shape, dtype=bool)
    watershed_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(watershed_mask)
    labels = watershed(-distance, markers, mask=final_mask)
    labels[~final_mask] = 0
    plt.imshow(labels, cmap=plt.cm.nipy_spectral)
    plt.show()

    if verbose:
        print(f"Found {len(np.unique(labels)) - 1} tumors")

    final_mask = (filters.sobel(labels) > 0)

    # assign
    st_adata.obs["tumor_boundary"] = "False"
    st_adata.obs["tumor_id"] = "non-tumor"
    for count_index, x, y in tqdm(
        zip(range(len(x_inds)), x_inds, y_inds), total=len(x_inds)
    ):
        # flip axes
        boundary_val = final_mask[y - 1, x - 1]
        tumor_id = labels[y - 1, x - 1]

        cellbc = st_adata.obs_names[count_index]

        st_adata.obs.loc[cellbc, "tumor_boundary"] = str(boundary_val)
        if tumor_id > 0:
            st_adata.obs.loc[cellbc, "tumor_id"] = f"Tumor-{tumor_id}"

    return st_adata

def plot_boundary(adata,
            polygon = None,
            ax=None,
            color='black',
            tumor_id = None,
            max_distance=50000,
	    width=1):

    def get_distance(x, y):
        
        return np.sum( (x - y)**2 )
    
    if tumor_id is not None:
        adata = adata[adata.obs['tumor_id'] == tumor_id]

    if not polygon:
        # create a series of shapely points
        tumor_boundary_points = adata.obsm['spatial'][adata.obs['tumor_boundary'] == 'True']

        # order points into circle
        all_points = tumor_boundary_points.copy()
        curr = Point(tumor_boundary_points[0,:])
        start = Point(tumor_boundary_points[0,:])
        point_list = [curr]

        # remove first element
        all_points = np.concatenate([all_points[:0],all_points[1:]])

        while len(all_points) > 1:

            # find closest point
            _next = np.argmin([get_distance(np.array([curr.x, curr.y]), all_points[i,:]) for i in range(len(all_points))])            
            distance_val = get_distance( np.array([curr.x, curr.y]), all_points[_next,:])

            if distance_val < max_distance:

                next_point = Point(all_points[_next,:])
                point_list.append(next_point)
                curr = next_point

            # remove next index and all ties from master list
            remove = True
            while remove and len(all_points) > 1:
                all_points = np.concatenate([all_points[:_next], all_points[_next+1:]])

                to_remove = [i for i in np.arange(len(all_points)) if get_distance(np.array([curr.x, curr.y]), all_points[i,:]) == distance_val]
                if len(to_remove) > 0:
                    _next = to_remove[0]
                else:
                    remove=False

        point_list.append(start)
        
        gdf1 = GeoDataFrame(point_list, columns = ['geometry'])    
        
        poly = Polygon([[p.x, p.y] for p in gdf1.geometry])
    else:
        poly = polygon
        
    gdf = GeoDataFrame([poly], columns=['geometry'])
    gdf.set_geometry('geometry')
    gdf.boundary.plot(ax=ax, color=color, linewidth=width)
