import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
import os

st.title("Category Similarity & Merging")

# 1. Upload files or load demo data
st.markdown('---')
demo_files = [
    os.path.join("Upload", "source1.csv"),
    os.path.join("Upload", "source2.csv"),
    os.path.join("Upload", "source3.csv"),
    os.path.join("Upload", "source4.csv")
]
col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True, key="uploader")
with col2:
    use_demo = st.session_state.get('use_demo', False)
    if st.button("Load demo data from server"):
        st.session_state['use_demo'] = True
        use_demo = True

if st.session_state.get('use_demo', False):
    def safe_read_csv(filepath):
        try:
            return pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(filepath, encoding='cp1251')
    dfs = [safe_read_csv(f) for f in demo_files]
    selected_dfs = []
    for i, df_demo in enumerate(dfs):
        df_demo['source_file'] = os.path.basename(demo_files[i])
        st.markdown(f"**Demo file: {os.path.basename(demo_files[i])}**")
        st.dataframe(df_demo)        
        columns = list(df_demo.columns)
        # Try to auto-detect columns  
        def find_col(targets):
            for t in targets:
                for c in columns:
                    if c.strip().lower() == t:
                        return c
            return ''
        # Also match 'Product Name' to 'product_name' for standardization
        product_guess = find_col(['product_name', 'product', 'name', 'product name'])
        category_guess = find_col(['category', 'cat'])
        sku_guess = find_col(['sku', 'код', 'id'])
        col1, col2, col3 = st.columns(3)
        with col1:
            col_product = st.selectbox(f"product name {os.path.basename(demo_files[i])}", [''] + columns, index=([''] + columns).index(product_guess) if product_guess in columns else 0, key=f"product_{i}")
        with col2:
            col_category = st.selectbox(f"category {os.path.basename(demo_files[i])}", [''] + columns, index=([''] + columns).index(category_guess) if category_guess in columns else 0, key=f"category_{i}")
        with col3:
            col_sku = st.selectbox(f"SKU column {os.path.basename(demo_files[i])}", [''] + columns, index=([''] + columns).index(sku_guess) if sku_guess in columns else 0, key=f"sku_{i}")
        # Standardize columns if selected
        rename_dict = {}
        # Always map 'Product Name' to 'product_name' for matching
        if col_product:
            if col_product.lower().strip() == 'product name':
                rename_dict[col_product] = 'product_name'
            else:
                rename_dict[col_product] = 'product_name'
        if col_category: rename_dict[col_category] = 'category'
        if col_sku: rename_dict[col_sku] = 'SKU'
        df_selected = df_demo.rename(columns=rename_dict)
        # If SKU not selected or not found, create it from product_name
        if 'SKU' not in df_selected.columns or not col_sku:
            if 'product_name' in df_selected.columns:
                df_selected['SKU'] = df_selected['product_name']
        selected_dfs.append(df_selected)
    df = pd.concat(selected_dfs, ignore_index=True)
    # Always add group_name column at start
    if 'group_name' not in df.columns:
        df['group_name'] = df['category'] if 'category' in df.columns else df.iloc[:,0]
    st.success("Demo data loaded and columns standardized.")
    st.dataframe(df)  # Show loaded demo table view
    # Reset demo flag if user uploads files
    if uploaded_files:
        st.session_state['use_demo'] = False
elif uploaded_files:
    # Add a column with the uploaded source file name after 'product_name'
    dfs = []
    def safe_read_csv_filelike(f):
        try:
            return pd.read_csv(f, encoding='utf-8')
        except UnicodeDecodeError:
            f.seek(0)
            return pd.read_csv(f, encoding='cp1251')
    for f in uploaded_files:
        temp_df = safe_read_csv_filelike(f)
        source_name = getattr(f, 'name', 'uploaded_file')
        insert_idx = temp_df.columns.get_loc('product_name') + 1 if 'product_name' in temp_df.columns else len(temp_df.columns)
        temp_df.insert(insert_idx, 'source_file', source_name)
        st.markdown(f"**Uploaded file: {source_name}**")
        st.dataframe(temp_df)
        columns = list(temp_df.columns)
        def find_col(targets):
            for t in targets:
                for c in columns:
                    if c.strip().lower() == t:
                        return c
            return ''
        product_guess = find_col(['product_name', 'product', 'name', 'product name'])
        category_guess = find_col(['category', 'cat'])
        sku_guess = find_col(['sku', 'код', 'id'])
        col1, col2, col3 = st.columns(3)
        with col1:
            col_product = st.selectbox(f"product name {source_name}", [''] + columns, index=([''] + columns).index(product_guess) if product_guess in columns else 0, key=f"product_up_{source_name}")
        with col2:
            col_category = st.selectbox(f"category  {source_name}", [''] + columns, index=([''] + columns).index(category_guess) if category_guess in columns else 0, key=f"category_up_{source_name}")
        with col3:
            col_sku = st.selectbox(f"SKU column for {source_name}", [''] + columns, index=([''] + columns).index(sku_guess) if sku_guess in columns else 0, key=f"sku_up_{source_name}")
        rename_dict = {}
        # Always map 'Product Name' to 'product_name' for matching
        if col_product:
            if col_product.lower().strip() == 'product name':
                rename_dict[col_product] = 'product_name'
            else:
                rename_dict[col_product] = 'product_name'
        if col_category: rename_dict[col_category] = 'category'
        if col_sku: rename_dict[col_sku] = 'SKU'
        temp_df = temp_df.rename(columns=rename_dict)
        # If SKU not selected or not found, create it from product_name
        if 'SKU' not in temp_df.columns or not col_sku:
            if 'product_name' in temp_df.columns:
                temp_df['SKU'] = temp_df['product_name']
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    # Always add group_name column at start
    if 'group_name' not in df.columns:
        df['group_name'] = df['category'] if 'category' in df.columns else df.iloc[:,0]
    st.success("Uploaded files loaded and columns standardized.")
    st.dataframe(df)
else:
    df = None

if df is not None:
    # Панель настроек
    st.markdown("### Панель настроек")
    group_only_diff_sources = st.checkbox(
        "Группировать категории только если они из разных источников",
        value=True
    )

    # 2. Select category column (only columns with 'category' in name are suggested)
    category_candidates = [col for col in df.columns if 'category' in col.lower()]
    if category_candidates:
        category_col = st.selectbox("Select the category column", category_candidates)
    else:
        category_col = st.selectbox("Select the category column", df.columns)

    # 3. Select embedding model (add Record Linkage as an option)
    model_name = st.selectbox(
        "Select embedding model or Record Linkage method",
        ["sentence-transformers/distiluse-base-multilingual-cased", "Record Linkage"]
    )
    if model_name == "sentence-transformers/distiluse-base-multilingual-cased":
        try:
            # --- Iterative clustering: use current_df for all rounds ---
            if 'current_df' not in st.session_state:
                st.session_state['current_df'] = df.copy()
            if 'fixed_clusters' not in st.session_state:
                st.session_state['fixed_clusters'] = set()
            if 'manual_join_selected' not in st.session_state:
                st.session_state['manual_join_selected'] = set()

            current_df = st.session_state['current_df']
            fixed_clusters = st.session_state['fixed_clusters']
            manual_join_selected = st.session_state['manual_join_selected']


            # --- Exclude fixed clusters from clustering ---
            # 1. Split current_df into fixed and unfixed parts
            fixed_group_names = st.session_state.get('fixed_group_names', set())
            if 'group_name' in current_df.columns:
                fixed_mask = current_df['group_name'].isin(fixed_group_names)
                unfixed_df = current_df[~fixed_mask].copy()
                fixed_df = current_df[fixed_mask].copy()
                categories = unfixed_df['group_name'].astype(str).unique()
            else:
                fixed_mask = current_df[category_col].isin(fixed_group_names)
                unfixed_df = current_df[~fixed_mask].copy()
                fixed_df = current_df[fixed_mask].copy()
                categories = unfixed_df[category_col].astype(str).unique()

            with st.spinner("Loading model and generating embeddings. This may take a few minutes on first run..."):
                model = SentenceTransformer(model_name, device="cpu")
                embeddings = model.encode(categories, show_progress_bar=False)
                st.write("Embeddings shape:", np.array(embeddings).shape)

            eps = st.slider("DBSCAN eps (distance threshold)", 0.1, 1.0, 0.4, 0.05, key="eps_main")
            min_samples = st.slider("DBSCAN min_samples", 1, 5, 2, key="min_samples_main")
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
            df_clusters = pd.DataFrame({
                "original_category": categories,
                "cluster": clustering.labels_
            })

            st.markdown("#### Review clusters and select those you want to join (rename all to first):")
            # Show fixed clusters first
            if not fixed_df.empty:
                st.markdown("---")
                st.markdown("#### Fixed clusters (excluded from further clustering):")
                for fixed_val in sorted(fixed_df['group_name'].unique() if 'group_name' in fixed_df.columns else fixed_df[category_col].unique()):
                    cluster_cats = fixed_df[fixed_df['group_name'] == fixed_val]['group_name'].unique() if 'group_name' in fixed_df.columns else fixed_df[fixed_df[category_col] == fixed_val][category_col].unique()
                    with st.expander(f"Fixed cluster: {fixed_val}"):
                        st.success("Fixed cluster")
                        st.success(list(cluster_cats))

            # Now show clusters for manual join (only for unfixed)
            for cluster_id in sorted(df_clusters["cluster"].unique()):
                cluster_key = f"cluster_{cluster_id}_fixed"
                is_fixed = cluster_id in fixed_clusters
                cluster_cats = df_clusters[df_clusters["cluster"] == cluster_id]["original_category"].tolist()
                with st.expander(f"Cluster {cluster_id}"):
                    st.json(cluster_cats)
                    checked = st.checkbox(f"Join this cluster (rename all to '{cluster_cats[0]}')", key=f"join_checkbox_{cluster_id}", value=(cluster_id in manual_join_selected))
                    if checked:
                        manual_join_selected.add(cluster_id)
                    else:
                        manual_join_selected.discard(cluster_id)

            colA, colB = st.columns([1,1])
            apply_btn = colA.button("Apply Manual Joins", key="apply_manual_joins_batch")
            rerun_btn = colB.button("Re-run clustering on current table", key="rerun_clustering")

            if apply_btn:
                renames = {}
                fixed_group_names = set()
                for cluster_id in manual_join_selected:
                    cluster_cats = df_clusters[df_clusters["cluster"] == cluster_id]["original_category"].tolist()
                    group_name = cluster_cats[0]
                    for cat in cluster_cats:
                        renames[cat] = group_name
                        fixed_group_names.add(group_name)
                # Сохраняем зафиксированные group_name в session_state
                st.session_state['fixed_group_names'] = st.session_state.get('fixed_group_names', set()) | fixed_group_names
                # Remove group_name if exists to avoid merge conflicts
                if 'group_name' in current_df.columns:
                    current_df = current_df.drop(columns=['group_name'])
                group_table = df_clusters.copy()
                group_table["group_name"] = group_table["original_category"].map(lambda x: renames.get(x, x))
                new_df = current_df.merge(group_table[["original_category", "group_name"]], left_on=(category_col), right_on="original_category", how="left")
                # Only filter by source_file if group_name was actually created
                if group_only_diff_sources and "source_file" in new_df.columns and "group_name" in new_df.columns:
                    new_df['source_file_count'] = new_df.groupby('group_name')['source_file'].transform('nunique')
                    new_df = new_df[new_df['source_file_count'] > 1].drop(columns=['source_file_count'])
                if "original_category" in new_df.columns:
                    new_df = new_df.drop(columns=["original_category"])
                st.session_state['current_df'] = new_df
                st.session_state['manual_join_selected'] = set()
                st.success("Manual joins applied. You can now re-run clustering on the updated table.")
                # Show only key columns in intermediate table
                show_cols = [col for col in ['group_name', 'category', 'product_name', 'SKU', 'source_file'] if col in new_df.columns]
                st.dataframe(new_df[show_cols])

            if rerun_btn:
                st.session_state['manual_join_selected'] = set()
                st.rerun()
            # 8. Fix groups as parent
            if st.button("Fix groups as parent"):
                group_table = df_clusters.copy()
                group_table["group_name"] = group_table.groupby("cluster")['original_category'].transform('first')
                grouped_df = df.merge(group_table[["original_category", "group_name"]], left_on=category_col, right_on="original_category", how="left")
                # Если group_name не был создан (все значения NaN), создаём его из category
                if "group_name" not in grouped_df.columns or grouped_df["group_name"].isnull().all():
                    grouped_df["group_name"] = grouped_df[category_col]
                # Если есть group_name_y, используем его как основной group_name
                if "group_name_y" in grouped_df.columns:
                    grouped_df["group_name"] = grouped_df["group_name_y"].combine_first(grouped_df["group_name_x"] if "group_name_x" in grouped_df.columns else grouped_df[category_col])
                # Удаляем только служебные столбцы, если есть
                for col in ["original_category", "group_name_x", "group_name_y"]:
                    if col in grouped_df.columns:
                        grouped_df = grouped_df.drop(columns=[col])
                # Предпросмотр финальной таблицы (первые 20 строк)
                st.markdown("#### Предпросмотр финального файла (первые 20 строк):")
                st.dataframe(grouped_df.head(20))
                # Кнопка скачивания
                st.download_button(
                    "Download grouped table CSV",
                    grouped_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    "grouped_categories.csv",
                    "text/csv"
                )
                import os
                save_path = os.path.join(os.getcwd(), "grouped_categories.csv")
                grouped_df.to_csv(save_path, index=False, encoding="utf-8-sig")
                st.info(f"Grouped table automatically saved to: {save_path}")
        except NotImplementedError as e:
            st.error(f"Model loading failed: {e}\n\n"
                     "This error may be caused by a mismatch between PyTorch and your hardware. "
                     "Try updating PyTorch, or running on a different machine/environment.")
    elif model_name == "Record Linkage":
        st.markdown("---")
        st.header("🔗 Record Linkage (Manual/Rule-based)")
        threshold = st.slider("Similarity threshold (%)", 0, 100, 50, 1)
        method = st.selectbox(
            "Choose record linkage method:",
            ["Name Similarity (RapidFuzz)", "Product Overlap", "Hybrid (Name + Product)"]
        )
        # Option to select column for value overlap (for Product Overlap/Hybrid)
        value_col = None
        if method in ["Product Overlap", "Hybrid (Name + Product)"]:
            value_candidates = [col for col in df.columns if col != category_col]
            if value_candidates:
                value_col = st.selectbox("Select column for value overlap (e.g., product name)", value_candidates)
            else:
                st.warning("No suitable columns found for value overlap.")
        categories = df[category_col].astype(str).unique()
        from itertools import combinations
        cat_pairs = list(combinations(categories, 2))
        sim_scores = []
        for cat1, cat2 in cat_pairs:
            score = 0
            details = ""
            if method == "Name Similarity (RapidFuzz)":
                score = fuzz.token_sort_ratio(cat1, cat2)
                details = f"Name similarity: {score:.1f}"
            elif method == "Product Overlap" and value_col:
                vals1 = set(df[df[category_col] == cat1][value_col].astype(str))
                vals2 = set(df[df[category_col] == cat2][value_col].astype(str))
                if vals1 or vals2:
                    overlap = len(vals1 & vals2) / max(1, len(vals1 | vals2))
                else:
                    overlap = 0
                score = overlap * 100
                details = f"Value overlap: {score:.1f}%"
            elif method == "Hybrid (Name + Product)" and value_col:
                name_score = fuzz.token_sort_ratio(cat1, cat2)
                vals1 = set(df[df[category_col] == cat1][value_col].astype(str))
                vals2 = set(df[df[category_col] == cat2][value_col].astype(str))
                if vals1 or vals2:
                    overlap = len(vals1 & vals2) / max(1, len(vals1 | vals2))
                else:
                    overlap = 0
                score = 0.7 * name_score + 0.3 * (overlap * 100)
                details = f"Hybrid: Name={name_score:.1f}, Overlap={overlap*100:.1f}"
            sim_scores.append({
                "Category 1": cat1,
                "Category 2": cat2,
                "Score": score,
                "Details": details
            })
        sim_df = pd.DataFrame(sim_scores)
        sim_df_filtered = sim_df[sim_df["Score"] >= threshold]
        st.dataframe(sim_df_filtered.sort_values("Score", ascending=False).reset_index(drop=True))
        st.download_button(
            "Download linkage scores CSV",
            sim_df_filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            "category_linkage_scores.csv",
            "text/csv"
        )
        # --- Final grouping table logic ---
        if st.button("Fix groups as parent (Record Linkage)"):
            # Build group mapping: each group is a set of categories linked by threshold
            import networkx as nx
            G = nx.Graph()
            for _, row in sim_df_filtered.iterrows():
                G.add_edge(row['Category 1'], row['Category 2'])
            # Each connected component is a group
            group_map = {}
            for i, comp in enumerate(nx.connected_components(G)):
                group_name = sorted(list(comp))[0]  # Use first category as group name
                for cat in comp:
                    group_map[cat] = group_name
            # Categories not in any group get their own group
            for cat in categories:
                if cat not in group_map:
                    group_map[cat] = cat
            grouped_df = df.copy()
            grouped_df['group_name'] = grouped_df[category_col].map(group_map)
            st.dataframe(grouped_df)
            st.download_button(
                "Download grouped table CSV",
                grouped_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                "grouped_categories.csv",
                "text/csv"
            )
            save_path = os.path.join(os.getcwd(), "grouped_categories.csv")
            grouped_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            st.info(f"Grouped table automatically saved to: {save_path}")
else:
    st.info("Please upload at least one CSV file.")

st.markdown("---")
st.markdown("**Instructions:** Upload CSVs, select the category column, choose an embedding model, adjust clustering, and download the mapping.")
