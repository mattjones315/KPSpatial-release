import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import typer
from collections import defaultdict

app = typer.Typer()


@app.command()
def prepare_infercnv_inputs(
        adata_path: str = typer.Argument(
            ..., help='Path to AnnData'
        ),
        sample_name: str = typer.Argument(
            ..., help="Sample name"
        ),
        gene_ordering_file: str = typer.Argument(
            ..., help='Path to gene ordering'
        ),
        output_directory: str = typer.Argument(
            ..., help='Path to output directory'
        ),
        bin_size: int = typer.Option(
            0, help="Number of spots to pool together."
        ),
):

    adata_raw = sc.read_h5ad(adata_path)
    adata_raw.var_names = adata_raw.var_names.astype('str')
    adata_raw.var_names_make_unique()

    gene_ordering = pd.read_csv(gene_ordering_file, sep='\t', header=None)
    gene_names = gene_ordering.iloc[:,0]

    if bin_size > 0:
        pooled_adata = pool_spots(adata_raw, bin_size=bin_size)
        counts = pooled_adata.to_df().T
        annotations = pd.DataFrame(pooled_adata.obs['annotation'])
        annotations.columns = ['tumor']

    else:
        counts = adata_raw.to_df().T
        annotations = pd.DataFrame(adata_raw.obs['tumor_id'])

    counts = counts.loc[np.intersect1d(gene_names, counts.index)]
    
    counts.to_csv(f'{output_directory}/infercnv/input/{sample_name}.counts.tsv', sep='\t')
    annotations.to_csv(f'{output_directory}/infercnv/input/{sample_name}.annotations.tsv', sep='\t', header=False)

def pool_spots(adata, bin_size=50):

    spatial_coords = adata.obsm["spatial"]

    x_bins = np.arange(0, np.max(spatial_coords[:, 0]), step=bin_size).astype(
        int
    )
    y_bins = np.arange(0, np.max(spatial_coords[:, 1]), step=bin_size).astype(
        int
    )

    pooled_spot_id = np.zeros((len(x_bins), len(y_bins))).astype(int)
    for x in range(pooled_spot_id.shape[0]):
        for y in range(pooled_spot_id.shape[1]):
            pooled_spot_id[x, y] = (y % pooled_spot_id.shape[1]) + (x * pooled_spot_id.shape[1])
                                                                    
    # bin counts
    x_inds = np.digitize(spatial_coords[:, 0], x_bins)
    y_inds = np.digitize(spatial_coords[:, 1], y_bins)

    pooled_X = np.zeros((len(x_bins)*len(y_bins), adata.X.shape[1]))
    pooled_coordinates = np.zeros((len(x_bins)*len(y_bins), 2))

    total_spots_in_pool = defaultdict(int)
    total_tumor_in_pool = defaultdict(int)
    # pooled_annotations = pd.DataFrame(index=[f'spot_{x}' for x in range(len(pooled_coordinates))], columns = ['total_spots', 'n_tumor'])

    # pool together
    for spot_index, x, y in zip(range(len(x_inds)), x_inds, y_inds):

        vals = np.array(adata.X[spot_index,:].todense())[0,:]
        spot_name = adata.obs_names[spot_index]
        is_tumor = (adata.obs.loc[spot_name, 'tumor_id'] != 'non-tumor')
        
        new_spot_id = pooled_spot_id[x-1, y-1]
        pooled_X[new_spot_id, :] += vals
        
        pooled_coordinates[new_spot_id, 0] = x_bins[x-1]
        pooled_coordinates[new_spot_id, 1] = y_bins[y-1]

        total_spots_in_pool[str(new_spot_id)] += 1
        total_tumor_in_pool[str(new_spot_id)] += int(is_tumor)

        # pooled_annotations.loc[f'spot_{new_spot_id}', 'total_spots'] += 1
        # pooled_annotations.loc[f'spot_{new_spot_id}', 'n_tumor'] += is_tumor


    pooled_adata = anndata.AnnData(pooled_X, obsm={"spatial": pooled_coordinates})
    sc.pp.calculate_qc_metrics(pooled_adata, inplace=True)

    pooled_adata.obs['total_spots'] = pooled_adata.obs_names.map(total_spots_in_pool)
    pooled_adata.obs['total_tumor'] = pooled_adata.obs_names.map(total_tumor_in_pool)
    pooled_adata.obs['tumor_frac'] = pooled_adata.obs.apply(lambda x: x.total_tumor / x.total_spots, axis=1)
    pooled_adata.obs['annotation'] = pooled_adata.obs.apply(lambda x: 'Tumor' if x.tumor_frac > 0.5 else 'Normal', axis=1)

    pooled_adata.var_names = adata.var_names

    sc.pp.filter_cells(pooled_adata, min_counts=1000)

    n_cells = pooled_adata.n_obs
    mean_rna = pooled_adata.obs["total_counts"].mean()
    print(
            f"Detected {n_cells} spots. Detected a mean UMI count of {mean_rna}."
        )

    pooled_adata.raw = pooled_adata

    # sc.pp.normalize_total(pooled_adata, target_sum=1e6)
    # sc.pp.log1p(pooled_adata)

    return pooled_adata

if __name__ == "__main__":
    app()
