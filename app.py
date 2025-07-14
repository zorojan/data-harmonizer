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

def robust_read_csv(file_or_path):
    """
    More robust CSV reading function that tries different encodings,
    delimiters, and handles bad lines to prevent ParserError.
    """
    # For file-like objects from upload, we need to be able to seek
    is_file_like = hasattr(file_or_path, 'seek')

    # List of configurations to try
    configs = [
        {'encoding': 'utf-8', 'sep': ',', 'on_bad_lines': 'warn'},
        {'encoding': 'utf-8', 'sep': ';', 'on_bad_lines': 'warn'},
        {'encoding': 'cp1251', 'sep': ',', 'on_bad_lines': 'warn'},
        {'encoding': 'cp1251', 'sep': ';', 'on_bad_lines': 'warn'},
        # Final attempt with python engine which is more robust
        {'encoding': 'utf-8', 'sep': None, 'engine': 'python', 'on_bad_lines': 'skip'}
    ]

    for config in configs:
        try:
            if is_file_like:
                file_or_path.seek(0)
            return pd.read_csv(file_or_path, **config)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
            continue # Try next configuration

    # If all attempts fail, raise an error
    st.error(f"Fatal Error: Could not parse the CSV file: {getattr(file_or_path, 'name', file_or_path)}. The file may be corrupted or in an unsupported format.")
    return None


col1, col2 = st.columns(2)
with col1:
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True, key="uploader")
with col2:
    use_demo = st.session_state.get('use_demo', False)
    if st.button("Load demo data from server"):
        st.session_state['use_demo'] = True
        use_demo = True

if st.session_state.get('use_demo', False):
    dfs = [robust_read_csv(f) for f in demo_files]
    # Filter out None results if a file failed to load
    dfs = [df for df in dfs if df is not None]
    if not dfs:
        st.stop() # Stop if no demo files could be loaded

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
    dfs = [robust_read_csv(f) for f in uploaded_files]
    # Filter out None results if a file failed to load
    dfs = [df for df in dfs if df is not None]
    if not dfs:
        st.stop() # Stop if no files could be loaded

    def safe_read_csv_filelike(f):
        try:
            return pd.read_csv(f, encoding='utf-8')
        except UnicodeDecodeError:
            f.seek(0)
            return pd.read_csv(f, encoding='cp1251')
    for i, temp_df in enumerate(dfs):
        source_name = getattr(uploaded_files[i], 'name', f'uploaded_file_{i}')
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
        # This line is removed as dfs is already a list of dataframes
        # dfs.append(temp_df)
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
        # --- Manual merge block: объединить любые категории вручную ---
        st.markdown("---")
        st.markdown("#### Manual merge: объединить любые категории вручную")
        # 1. Исключить уже зафиксированные категории из списка для объединения
        fixed_group_names_set = st.session_state.get('fixed_group_names', set())
        manual_fixed_assignments = st.session_state.get('manual_fixed_assignments', {})
        # Собираем все зафиксированные категории
        fixed_cats = set()
        for group, cats in manual_fixed_assignments.items():
            fixed_cats.update(cats)
        # Формируем список для выбора: если категория зафиксирована, показываем с пометкой
        all_categories = sorted(df[category_col].astype(str).unique())
        manual_merge_options = []
        manual_merge_labels = {}
        for cat in all_categories:
            fixed_group = None
            for group, cats in manual_fixed_assignments.items():
                if cat in cats:
                    fixed_group = group
                    break
            if fixed_group:
                manual_merge_labels[cat] = f"{cat} (уже в группе '{fixed_group}')"
            else:
                manual_merge_options.append(cat)
                manual_merge_labels[cat] = cat
        manual_merge_cats = st.multiselect(
            "Выберите категории для объединения вручную",
            options=manual_merge_options,
            format_func=lambda x: manual_merge_labels[x],
            key="manual_merge_cats"
        )
        manual_merge_name = st.text_input("Новое имя группы для выбранных категорий", key="manual_merge_name")
        if st.button("Применить ручное объединение"):
            if manual_merge_cats and manual_merge_name:
                # Проверка: не дать одной категории попасть в несколько групп
                overlap = set(manual_merge_cats) & fixed_cats
                if overlap:
                    st.error(f"Категории {list(overlap)} уже зафиксированы в других группах. Снимите их из выбора.")
                else:
                    # Обновить group_name для выбранных категорий в current_df
                    current_df = st.session_state['current_df']
                    current_df.loc[current_df[category_col].isin(manual_merge_cats), 'group_name'] = manual_merge_name
                    st.session_state['current_df'] = current_df
                    # Добавить в историю ручных объединений
                    if manual_merge_name not in manual_fixed_assignments:
                        manual_fixed_assignments[manual_merge_name] = set()
                    manual_fixed_assignments[manual_merge_name].update(manual_merge_cats)
                    st.session_state['manual_fixed_assignments'] = manual_fixed_assignments
                    # Добавить в фиксированные группы
                    fixed_group_names = st.session_state.get('fixed_group_names', set())
                    fixed_group_names.add(manual_merge_name)
                    st.session_state['fixed_group_names'] = fixed_group_names
                    st.success(f"Категории {manual_merge_cats} объединены в группу '{manual_merge_name}'")
            else:
                st.warning("Выберите категории и введите имя группы.")
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

            if len(categories) == 0:
                st.warning("No more categories left for clustering. All clusters are fixed.")
                df_clusters = pd.DataFrame({"original_category": [], "cluster": []})
            else:
                with st.spinner("Loading model and generating embeddings. This may take a few minutes on first run..."):
                    import torch
                    # Try to force CPU and handle meta tensor error
                    try:
                        model = SentenceTransformer(model_name, device="cpu")
                        # Check for meta tensors and reload if needed
                        for name, param in model.named_parameters():
                            if hasattr(param, 'is_meta') and param.is_meta:
                                st.warning(f"Parameter {name} is a meta tensor. Attempting to reload on CPU...")
                                model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
                                break
                        embeddings = model.encode(categories, show_progress_bar=False)
                        st.write("Embeddings shape:", np.array(embeddings).shape)
                    except RuntimeError as e:
                        st.error(f"PyTorch RuntimeError: {e}\n\nПопробуйте обновить PyTorch и sentence-transformers, либо используйте только CPU-окружение.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Model loading failed: {e}\n\nThis error may be caused by a mismatch between PyTorch and your hardware. Try updating PyTorch, or running on a different machine/environment.")
                        st.stop()

                eps = st.slider("DBSCAN eps (distance threshold)", 0.1, 1.0, 0.4, 0.05, key="eps_main")
                min_samples = st.slider("DBSCAN min_samples", 1, 5, 2, key="min_samples_main")
                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
                df_clusters = pd.DataFrame({
                    "original_category": categories,
                    "cluster": clustering.labels_
                })

            st.markdown("#### Review clusters and select those you want to join (rename all to first):")
            # Show all fixed clusters from all rounds (not just current fixed_df)
            fixed_group_names_all = st.session_state.get('fixed_group_names', set())
            if fixed_group_names_all:
                st.markdown("---")
                st.markdown("#### Fixed clusters (excluded from further clustering):")
                # Find all categories assigned to each fixed group_name in ALL previous rounds
                # We'll search in all DataFrames that ever had group_name assignments
                # For robustness, use both the original df and all session_state['manual_fixed_assignments'] if exists
                manual_fixed_assignments = st.session_state.get('manual_fixed_assignments', {})
                for fixed_val in sorted(fixed_group_names_all):
                    assigned_cats = set()
                    # 1. From manual_fixed_assignments (if exists)
                    if fixed_val in manual_fixed_assignments:
                        assigned_cats.update(manual_fixed_assignments[fixed_val])
                    # 2. From all current and previous DataFrames
                    for search_df in [st.session_state.get('current_df', df), df]:
                        if 'group_name' in search_df.columns:
                            assigned_cats.update(search_df[search_df['group_name'] == fixed_val][category_col].unique())
                    # Fallback: if nothing found, just show the group name itself
                    if not assigned_cats:
                        assigned_cats = {fixed_val}
                    with st.expander(f"Fixed cluster: {fixed_val}"):
                        st.success(list(assigned_cats))

            # Now show clusters for manual join (only for unfixed)
            for cluster_id in sorted(df_clusters["cluster"].unique()):
                cluster_key = f"cluster_{cluster_id}_fixed"
                is_fixed = cluster_id in fixed_clusters
                cluster_cats = df_clusters[df_clusters["cluster"] == cluster_id]["original_category"].tolist()
                is_ungrouped = (cluster_id == -1)
                expander_label = f"Cluster {'ungrouped' if is_ungrouped else cluster_id}"
                with st.expander(expander_label, expanded=not is_fixed):
                    if is_ungrouped:
                        st.warning("This cluster contains ungrouped categories. They will not be merged with others until you select and apply a join.")
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
                manual_fixed_assignments = st.session_state.get('manual_fixed_assignments', {})
                for cluster_id in manual_join_selected:
                    cluster_cats = df_clusters[df_clusters["cluster"] == cluster_id]["original_category"].tolist()
                    group_name = cluster_cats[0]
                    for cat in cluster_cats:
                        renames[cat] = group_name
                        fixed_group_names.add(group_name)
                        # Track manual assignments for history
                        if group_name not in manual_fixed_assignments:
                            manual_fixed_assignments[group_name] = set()
                        manual_fixed_assignments[group_name].add(cat)
                # Save manual assignments for all rounds
                st.session_state['manual_fixed_assignments'] = manual_fixed_assignments
                st.session_state['fixed_group_names'] = st.session_state.get('fixed_group_names', set()) | fixed_group_names
                # Remove group_name if exists to avoid merge conflicts
                if 'group_name' in current_df.columns:
                    current_df = current_df.drop(columns=['group_name'])
                group_table = df_clusters.copy()
                group_table["group_name"] = group_table["original_category"].map(lambda x: renames.get(x, x))
                new_df = current_df.merge(group_table[["original_category", "group_name"]], left_on=(category_col), right_on="original_category", how="left")
                # Only filter by source_file if group_name was actually created
                # FIX: Do not drop ungrouped categories (group_name == -1) when filtering by source_file
                if group_only_diff_sources and "source_file" in new_df.columns and "group_name" in new_df.columns:
                    new_df['source_file_count'] = new_df.groupby('group_name')['source_file'].transform('nunique')
                    # Keep all ungrouped (-1) rows, filter only grouped
                    mask_ungrouped = new_df['group_name'] == -1
                    mask_grouped = new_df['source_file_count'] > 1
                    new_df = new_df[mask_ungrouped | mask_grouped].drop(columns=['source_file_count'])
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

            # 8. Download final processed file (from current_df)
            st.markdown("---")
            st.markdown("#### Download the final processed table (all manual and automatic merges applied):")
            # Build a full final table: merge all fixed and unfixed rows from the original df
            # 1. Get all fixed group names and their members from session_state
            fixed_group_names = st.session_state.get('fixed_group_names', set())
            current_df = st.session_state.get('current_df', df)
            # Build fixed assignments from ALL rounds, not just current_df
            fixed_assignments = {}
            manual_fixed_assignments = st.session_state.get('manual_fixed_assignments', {})
            for fixed_group in fixed_group_names:
                # 1. From manual_fixed_assignments (if exists)
                if fixed_group in manual_fixed_assignments:
                    for cat in manual_fixed_assignments[fixed_group]:
                        fixed_assignments[cat] = fixed_group
                # 2. From all current and previous DataFrames
                for search_df in [st.session_state.get('current_df', df), df]:
                    if 'group_name' in search_df.columns:
                        matches = search_df[search_df['group_name'] == fixed_group]
                        for _, row in matches.iterrows():
                            fixed_assignments[row[category_col]] = fixed_group
            # 2. For all rows in the original df, assign group_name: fixed if exists, else from current_df, else itself
            def get_final_group(row):
                cat = row[category_col]
                # Priority: fixed assignment > current_df group_name > ungrouped marker
                if cat in fixed_assignments:
                    return fixed_assignments[cat]
                # Try to get from current_df (may be unfixed group)
                if 'group_name' in current_df.columns:
                    match = current_df[current_df[category_col] == cat]
                    if not match.empty:
                        val = match.iloc[0]['group_name']
                        # If group_name is not the same as the original category, treat as grouped
                        if val != cat:
                            return val
                # If not grouped, mark as ungrouped (use -1, or 'ungrouped')
                return 'ungrouped'
            final_df = df.copy()
            final_df['group_name'] = final_df.apply(get_final_group, axis=1)
            # Remove only service columns if present
            for col in ["original_category", "group_name_x", "group_name_y"]:
                if col in final_df.columns:
                    final_df = final_df.drop(columns=[col])
            st.dataframe(final_df.head(20))
            # Сохраняем финальный результат в grouped_categories.csv
            save_path = os.path.join(os.getcwd(), "grouped_categories.csv")
            try:
                final_df.to_csv(save_path, index=False, encoding="utf-8-sig")
                st.info(f"Финальный сгруппированный результат сохранён в: {save_path}")
            except Exception as e:
                st.warning(f"Не удалось сохранить grouped_categories.csv: {e}")
            st.download_button(
                "Download final grouped table CSV",
                final_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                "grouped_categories.csv",
                "text/csv"
            )
            # Remove auto-save to file to avoid PermissionError on Windows if file is open
            # import os
            # save_path = os.path.join(os.getcwd(), "grouped_categories.csv")
            # try:
            #     final_df.to_csv(save_path, index=False, encoding="utf-8-sig")
            #     st.info(f"Final grouped table automatically saved to: {save_path}")
            # except PermissionError:
            #     st.warning(f"Could not save to {save_path} (file may be open). Please close the file and try again.")
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




# --- Кнопка перехода на страницу оптимизации параметров ---

if df is not None:
    st.markdown("---")
    if st.button("Сохранить данные и перейти к оптимизации параметров", key="go_to_param_page"):
        # Сохраняем финальный результат в grouped_categories.csv
        save_path = os.path.join(os.getcwd(), "grouped_categories.csv")
        try:
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            st.info(f"Финальный результат сохранён в: {save_path}")
        except Exception as e:
            st.warning(f"Не удалось сохранить grouped_categories.csv: {e}")
        st.switch_page('pages/param_processing.py')

st.markdown("**Instructions:** Upload CSVs, select the category column, choose an embedding model, adjust clustering, and download the mapping.")
