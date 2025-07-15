import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
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
        
        with st.expander(f"📁 Demo file: {os.path.basename(demo_files[i])}", expanded=False):
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

    for i, temp_df in enumerate(dfs):
        source_name = getattr(uploaded_files[i], 'name', f'uploaded_file_{i}')
        insert_idx = temp_df.columns.get_loc('product_name') + 1 if 'product_name' in temp_df.columns else len(temp_df.columns)
        temp_df.insert(insert_idx, 'source_file', source_name)
        
        with st.expander(f"📁 Uploaded file: {source_name}", expanded=False):
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
        value=True,
        help="Если включено: объединяются только категории из разных файлов. Уникальные категории из одного источника остаются отдельными. Если отключено: группируются все похожие категории независимо от источника."
    )
    
    # Добавляем чекбокс для ручного режима
    manual_mode = st.checkbox(
        "Ручной режим обработки (manual mode)",
        value=False,
        help="В автоматическом режиме все кластеры (кроме ungrouped) объединяются автоматически. В ручном режиме вы можете выбирать какие кластеры объединять."
    )

    # 2. Select category column (only columns with 'category' in name are suggested)
    category_candidates = [col for col in df.columns if 'category' in col.lower()]
    if category_candidates:
        category_col = st.selectbox("Select the category column", category_candidates)
    else:
        category_col = st.selectbox("Select the category column", df.columns)

    # 3. Select embedding model
    model_name = st.selectbox(
        "Select embedding model",
        ["sentence-transformers/distiluse-base-multilingual-cased"]
    )
    
    if model_name == "sentence-transformers/distiluse-base-multilingual-cased":
        
        try:
            # --- Iterative clustering: use current_df for all rounds ---
            if 'current_df' not in st.session_state:
                st.session_state['current_df'] = df.copy()
            if 'manual_join_selected' not in st.session_state:
                st.session_state['manual_join_selected'] = set()
            if 'clusters_to_explode' not in st.session_state:
                st.session_state['clusters_to_explode'] = set()

            current_df = st.session_state['current_df']
            manual_join_selected = st.session_state['manual_join_selected']


            # --- Exclude fixed clusters from clustering ---
            # 1. Split current_df into fixed and unfixed parts
            fixed_group_names = st.session_state.get('fixed_group_names', set())
            if 'group_name' in current_df.columns:
                # Берем категории, которые НЕ зафиксированы
                fixed_mask = current_df['group_name'].isin(fixed_group_names)
                unfixed_df = current_df[~fixed_mask].copy()
                
                # ✅ ИСПРАВЛЕНИЕ: Берем исходные категории, не group_name
                categories = unfixed_df[category_col].astype(str).unique()
            else:
                fixed_mask = current_df[category_col].isin(fixed_group_names)
                unfixed_df = current_df[~fixed_mask].copy()
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

            # АВТОМАТИЧЕСКИЙ РЕЖИМ - объединяем все кластеры (кроме ungrouped) автоматически
            if not manual_mode and len(df_clusters) > 0:
                st.markdown("### 🤖 Автоматический режим активен")
                st.info("Все кластеры (кроме ungrouped) объединяются автоматически")
                st.info("💡 **Совет:** Включите 'Ручной режим обработки' выше, если хотите контролировать объединение кластеров или разбить уже созданные группы")
                
                # Автоматически применяем все кластеры кроме -1 (ungrouped)
                auto_renames = {}
                auto_fixed_group_names = set()
                manual_fixed_assignments = st.session_state.get('manual_fixed_assignments', {})
                
                for cluster_id in df_clusters["cluster"].unique():
                    if cluster_id != -1:  # Исключаем ungrouped
                        cluster_cats = df_clusters[df_clusters["cluster"] == cluster_id]["original_category"].tolist()
                        if len(cluster_cats) > 1:  # Только если в кластере больше одной категории
                            group_name = cluster_cats[0]
                            for cat in cluster_cats:
                                auto_renames[cat] = group_name
                                auto_fixed_group_names.add(group_name)
                                # Track manual assignments for history
                                if group_name not in manual_fixed_assignments:
                                    manual_fixed_assignments[group_name] = set()
                                manual_fixed_assignments[group_name].add(cat)
                
                if auto_renames:
                    # Save manual assignments for all rounds
                    st.session_state['manual_fixed_assignments'] = manual_fixed_assignments
                    st.session_state['fixed_group_names'] = st.session_state.get('fixed_group_names', set()) | auto_fixed_group_names
                    
                    # Remove group_name if exists to avoid merge conflicts
                    if 'group_name' in current_df.columns:
                        current_df = current_df.drop(columns=['group_name'])
                    group_table = df_clusters.copy()
                    group_table["group_name"] = group_table["original_category"].map(lambda x: auto_renames.get(x, x))
                    new_df = current_df.merge(group_table[["original_category", "group_name"]], left_on=(category_col), right_on="original_category", how="left")
                    
                    # Filter by source_file if needed
                    if group_only_diff_sources and "source_file" in new_df.columns and "group_name" in new_df.columns:
                        new_df['source_file_count'] = new_df.groupby('group_name')['source_file'].transform('nunique')
                        # ✅ ИСПРАВЛЕНИЕ: Правильная маска для ungrouped
                        mask_ungrouped = new_df['group_name'] == new_df[category_col]  # Уникальные категории
                        mask_grouped = new_df['source_file_count'] > 1  # Группы из разных источников
                        new_df = new_df[mask_ungrouped | mask_grouped].drop(columns=['source_file_count'])
                        
                        # Debug информация для диагностики
                        st.write(f"🔍 Фильтрация по источникам:")
                        st.write(f"- Уникальных категорий: {len(new_df[new_df['group_name'] == new_df[category_col]])}")
                        st.write(f"- Сгруппированных из разных источников: {len(new_df[new_df['group_name'] != new_df[category_col]])}")
                        
                    if "original_category" in new_df.columns:
                        new_df = new_df.drop(columns=["original_category"])
                    st.session_state['current_df'] = new_df
                    
                    st.success(f"Автоматически объединено {len(auto_fixed_group_names)} групп категорий")
                    # Show only key columns in intermediate table
                    show_cols = [col for col in ['group_name', 'category', 'product_name', 'SKU', 'source_file'] if col in new_df.columns]
                    st.dataframe(new_df[show_cols])
                else:
                    st.info("Нет кластеров для автоматического объединения")

            # РУЧНОЙ РЕЖИМ - показываем кластеры для выбора
            elif manual_mode:
                st.markdown("### 🔧 Ручной режим активен")
                st.markdown("#### Review clusters and select those you want to join (rename all to first):")
                
                # Show all fixed clusters from all rounds (not just current fixed_df)
                fixed_group_names_all = st.session_state.get('fixed_group_names', set())
                if fixed_group_names_all:
                    st.markdown("---")
                    st.markdown("#### Fixed clusters (excluded from further clustering):")
                    # Find all categories assigned to each fixed group_name in ALL previous rounds
                    manual_fixed_assignments = st.session_state.get('manual_fixed_assignments', {})
                    
                    # Track which clusters to explode
                    if 'clusters_to_explode' not in st.session_state:
                        st.session_state['clusters_to_explode'] = set()
                    clusters_to_explode = st.session_state['clusters_to_explode']
                    
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
                        
                        with st.expander(f"Fixed cluster: {fixed_val} ({len(assigned_cats)} категорий)"):
                            st.success(list(assigned_cats))
                            
                            # Добавляем чекбокс для разбиения кластера
                            explode_checked = st.checkbox(
                                f"🔄 Explode this cluster (разбить группу '{fixed_val}' на отдельные категории)",
                                key=f"explode_checkbox_{fixed_val}",
                                value=(fixed_val in clusters_to_explode),
                                help="Это разбивает группу обратно на отдельные категории для повторной кластеризации"
                            )
                            
                            if explode_checked:
                                clusters_to_explode.add(fixed_val)
                            else:
                                clusters_to_explode.discard(fixed_val)
                    
                    # Кнопка для применения разбиения кластеров
                    if clusters_to_explode:
                        st.markdown("---")
                        if st.button(f"🔄 Explode selected clusters ({len(clusters_to_explode)} выбрано)", key="explode_clusters_btn"):
                            # Разбиваем выбранные кластеры
                            manual_fixed_assignments = st.session_state.get('manual_fixed_assignments', {})
                            fixed_group_names = st.session_state.get('fixed_group_names', set())
                            current_df = st.session_state.get('current_df', df)
                            
                            for cluster_to_explode in clusters_to_explode:
                                # Удаляем из fixed_group_names
                                fixed_group_names.discard(cluster_to_explode)
                                
                                # Удаляем из manual_fixed_assignments
                                if cluster_to_explode in manual_fixed_assignments:
                                    del manual_fixed_assignments[cluster_to_explode]
                                
                                # В current_df возвращаем group_name обратно к исходным категориям
                                if 'group_name' in current_df.columns:
                                    mask = current_df['group_name'] == cluster_to_explode
                                    current_df.loc[mask, 'group_name'] = current_df.loc[mask, category_col]
                            
                            # Обновляем session_state
                            st.session_state['fixed_group_names'] = fixed_group_names
                            st.session_state['manual_fixed_assignments'] = manual_fixed_assignments
                            st.session_state['current_df'] = current_df
                            st.session_state['clusters_to_explode'] = set()  # Очищаем список
                            
                            st.success(f"Разбито {len(clusters_to_explode)} кластеров. Категории готовы для повторной кластеризации.")
                            st.rerun()

                # Now show clusters for manual join (only for unfixed)
                for cluster_id in sorted(df_clusters["cluster"].unique()):
                    cluster_cats = df_clusters[df_clusters["cluster"] == cluster_id]["original_category"].tolist()
                    is_ungrouped = (cluster_id == -1)
                    expander_label = f"Cluster {'ungrouped' if is_ungrouped else cluster_id}"
                    with st.expander(expander_label, expanded=True):
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
                reset_btn = colB.button("🔄 Reset All Groups & Re-cluster", key="reset_all_groups", 
                                      help="Сбросить все фиксированные группы и запустить автоматическую кластеризацию заново")

                if reset_btn:
                    # Сбрасываем все фиксированные группы и состояние
                    st.session_state['fixed_group_names'] = set()
                    st.session_state['manual_fixed_assignments'] = {}
                    st.session_state['current_df'] = df.copy()
                    st.session_state['manual_join_selected'] = set()
                    st.session_state['clusters_to_explode'] = set()
                    
                    st.success("Все группы сброшены! Запускается автоматическая кластеризация с текущими параметрами...")
                    st.rerun()

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
                    if group_only_diff_sources and "source_file" in new_df.columns and "group_name" in new_df.columns:
                        new_df['source_file_count'] = new_df.groupby('group_name')['source_file'].transform('nunique')
                        # ✅ ИСПРАВЛЕНИЕ: Правильная маска для ungrouped
                        mask_ungrouped = new_df['group_name'] == new_df[category_col]  # Уникальные категории
                        mask_grouped = new_df['source_file_count'] > 1  # Группы из разных источников
                        new_df = new_df[mask_ungrouped | mask_grouped].drop(columns=['source_file_count'])
                        
                        # Debug информация для диагностики  
                        st.write(f"🔍 Фильтрация по источникам в ручном режиме:")
                        st.write(f"- Уникальных категорий: {len(new_df[new_df['group_name'] == new_df[category_col]])}")
                        st.write(f"- Сгруппированных из разных источников: {len(new_df[new_df['group_name'] != new_df[category_col]])}")
                    if "original_category" in new_df.columns:
                        new_df = new_df.drop(columns=["original_category"])
                    st.session_state['current_df'] = new_df
                    st.session_state['manual_join_selected'] = set()
                    st.success("Manual joins applied. You can now re-run clustering on the updated table.")
                    # Show only key columns in intermediate table
                    show_cols = [col for col in ['group_name', 'category', 'product_name', 'SKU', 'source_file'] if col in new_df.columns]
                    st.dataframe(new_df[show_cols])

                # --- Manual merge block: показывать только в ручном режиме ---
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

            # Кнопка сброса для автоматического режима
            if not manual_mode:
                if st.button("🔄 Reset All Groups & Re-cluster", key="reset_all_groups_auto",
                           help="Сбросить все фиксированные группы и запустить автоматическую кластеризацию заново"):
                    # Сбрасываем все фиксированные группы и состояние
                    st.session_state['fixed_group_names'] = set()
                    st.session_state['manual_fixed_assignments'] = {}
                    st.session_state['current_df'] = df.copy()
                    st.session_state['manual_join_selected'] = set()
                    st.session_state['clusters_to_explode'] = set()
                    
                    st.success("Все группы сброшены! Запускается автоматическая кластеризация с текущими параметрами...")
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
            st.download_button(
                "Download final grouped table CSV",
                final_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                "final_grouped_table.csv",
                "text/csv"
            )
        except NotImplementedError as e:
            st.error(f"Model loading failed: {e}\n\n"
                     "This error may be caused by a mismatch between PyTorch and your hardware. "
                     "Try updating PyTorch, or running on a different machine/environment.")
        
        # Добавить после секции "Download the final processed table":
        st.markdown("---")
        st.markdown("#### 📊 Статистика группировки:")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_categories = len(df[category_col].unique())
            st.metric("Всего категорий", total_categories)
        with col2:
            grouped_categories = len([g for g in final_df['group_name'].unique() if g != 'ungrouped'])
            st.metric("Сгруппированных", grouped_categories)
        with col3:
            reduction_percent = round((1 - grouped_categories/total_categories) * 100, 1) if total_categories > 0 else 0
            st.metric("Сокращение", f"{reduction_percent}%")

        # Топ групп по размеру
        st.markdown("**Топ-5 самых больших групп:**")
        group_sizes = final_df[final_df['group_name'] != 'ungrouped']['group_name'].value_counts().head(5)
        st.bar_chart(group_sizes)
else:
    st.info("Please upload at least one CSV file.")

# --- Кнопка перехода на страницу оптимизации параметров ---

if df is not None:
    st.markdown("---")
    if st.button("Сохранить данные и перейти к оптимизации параметров", key="go_to_param_page"):
        # Строим финальную таблицу точно так же, как в секции "Download the final processed table"
        fixed_group_names = st.session_state.get('fixed_group_names', set())
        current_df = st.session_state.get('current_df', df)
        
        # Build fixed assignments from ALL rounds
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
        
        # For all rows in the original df, assign group_name: fixed if exists, else from current_df, else itself
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
            
        final_table = df.copy()
        final_table['group_name'] = final_table.apply(get_final_group, axis=1)
        # Remove only service columns if present
        for col in ["original_category", "group_name_x", "group_name_y"]:
            if col in final_table.columns:
                final_table = final_table.drop(columns=[col])
            
        # Сохраняем финальную обработанную таблицу в grouped_categories.csv (для перехода к оптимизации)
        save_path = os.path.join(os.getcwd(), "grouped_categories.csv")
        try:
            final_table.to_csv(save_path, index=False, encoding="utf-8-sig")
            st.info(f"Финальная обработанная таблица сохранена в: {save_path}")
        except Exception as e:
            st.warning(f"Не удалось сохранить grouped_categories.csv: {e}")
        st.switch_page('pages/param_processing.py')

st.markdown("**Instructions:** Upload CSVs, select the category column, choose an embedding model, adjust clustering, and download the mapping.")
