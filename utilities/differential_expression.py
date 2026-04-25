"""
Runs differential expression.
"""
from typing import Optional, Union

import anndata
import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import typer
from tqdm.auto import tqdm

app = typer.Typer()


@app.command()
def differential_expression(
    adata_path: str = typer.Argument(..., help="Path to anndata object."),
    cluster_labels: str = typer.Argument(..., help="Cluster label for DE"),
    adata: anndata.AnnData = typer.Option(None, help='Anndata object.'),
    output_path: Optional[str] = typer.Option(None, help='Path to output file.'),
    de_method: str = typer.Option("wilcoxon", help="DE method"),
    n_top_genes: int = typer.Option(2000, help='Number of top genes to find for each cluster.'),
    layer: int = typer.Option("logged", help="Layer of preprocessed data to use."),
) -> Union[pd.DataFrame, None]:

    if not adata:
        adata = sc.read_h5ad(adata_path)

    print(">> Normalizing...")
    adata_raw = adata.raw.to_adata()
    adata_raw.layers['counts'] = adata_raw.X.copy()
    adata_raw.layers['logged'] = adata_raw.X.copy()
    scale_factor = np.median(np.array(adata_raw.X.sum(axis=1)))
    sc.pp.normalize_total(adata_raw, target_sum=scale_factor, layer='logged')
    sc.pp.log1p(adata_raw, layer='logged')

    print(">> Running DE...")
    sc.tl.rank_genes_groups(adata_raw,
                            cluster_labels,
                            method=de_method,
                            layer = layer,
                            use_raw=False,
                            n_genes=n_top_genes,
                            pos_only=True)

    # calculate % expressed
    result = adata_raw.uns['rank_genes_groups']
    groups = result['names'].dtype.names

    print(">> Adding percent expressed...")
    result['percent_expressed'] = {}
    for group in tqdm(groups):
        
        cells = adata_raw.obs[adata_raw.obs[cluster_labels] == group].index.values
        names = result['names'][group]
        
        mask = adata_raw[:,names][cells,:].X.todense()
        mask[mask > 0] = 1
        
        sums = np.squeeze(np.asarray(np.sum(mask, axis=0)))
        result['percent_expressed'][group] = (sums / len(cells))
        
    result['percent_expressed'][groups[0]]

    print(">> Adding AUROC and specificity...")
    result['auroc'] = {}
    result['auprc'] = {}
    for group in tqdm(groups):
        
        labels = (adata_raw.obs[cluster_labels] == group).values
        
        name_list = result['names'][group]
        
        aurocs = []
        auprcs = []
        for gene in tqdm(name_list):
            vec=adata_raw[:,[gene]].X
            y_score = vec.todense()
            aurocs.append(metrics.roc_auc_score(np.asarray(labels), np.asarray(y_score)))
            auprcs.append(metrics.average_precision_score(np.asarray(labels), np.asarray(y_score)))
        result['auroc'][group] = aurocs
        result['auprc'][group] = auprcs

    result = adata_raw.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    de_res = pd.DataFrame(
        {'cluster' + group + '_' + key: result[key][group]
        for group in groups for key in ['names', 'logfoldchanges', 'pvals_adj', 'percent_expressed', 'auroc', 'auprc']})

    if output_path:
        de_res.to_csv(output_path, sep='\t')
    else:
        return de_res
    
if __name__ == "__main__":
    app()
