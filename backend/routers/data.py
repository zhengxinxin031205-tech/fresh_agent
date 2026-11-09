"""
数据管理路由
============
- 列出/选择内置示例数据集
- 上传用户自有 CSV / Excel 文件
- 查看数据质量画像与字段映射
- 上传历史记录:保存副本,一键重新加载之前上传过的数据
"""
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.core.data_adapter import DataAdapter, DataSession, STANDARD_FIELDS, SUPPLEMENT_FIELD_INFO, _fuzzy_match
from backend.core.analysis_service import AnalysisService
from backend.core.data_validator import validate_data

router = APIRouter()


def _require_loaded():
    """检查数据是否已加载"""
    svc = AnalysisService.get()
    if not svc.is_loaded():
        raise HTTPException(status_code=400, detail="尚未加载数据,请先上传数据或选择内置数据集")


def _uploads_dir() -> Path:
    """上传历史保存目录(项目 data/.uploads,打包后为exe同级 data/.uploads)"""
    import os
    env_dir = os.environ.get("FRESH_AGENT_DATA_DIR")
    if env_dir:
        return Path(env_dir) / ".uploads"
    # 兜底:开发模式下按代码位置推算项目根目录下的 data 目录
    return Path(__file__).resolve().parent.parent.parent / "data" / ".uploads"


def _tier_label(tier: str) -> str:
    """数据档次转商家可读名称"""
    return {"basic": "基础版", "advanced": "进阶版", "full": "完整版"}.get(tier, "基础版")


def _field_label(field: str) -> str:
    """标准字段转商家可读名称(补充合并结果提示用)"""
    info = SUPPLEMENT_FIELD_INFO.get(field)
    if info:
        return info["label"]
    aliases = STANDARD_FIELDS.get(field, [])
    return aliases[0] if aliases else field


def _pre_sample_raw(raw_df: pd.DataFrame):
    """在昂贵的标准化之前,对超大原始数据做轻量预采样

    只做列名识别(不碰数据内容),找到日期/商品/销量列后按时间+商品截取,
    避免对几百万行数据做完整清洗/排序/去重/画像后才发现只需要最近90天。
    返回 (截取后的raw_df, 面向商家的采样说明或None);小数据集原样返回。
    """
    MAX_ROWS = 1_000_000
    MAX_SKUS = 200
    MAX_DAYS = 90
    MAX_STORES = 20
    # 轻量列名识别:只看列名,不做数据清洗
    date_col = sku_col = sales_col = store_col = None
    for col in raw_df.columns:
        std = _fuzzy_match(col)
        if std == "date" and date_col is None:
            date_col = col
        elif std == "sku_id" and sku_col is None:
            sku_col = col
        elif std == "sales" and sales_col is None:
            sales_col = col
        elif std == "store_id" and store_col is None:
            store_col = col
    # 数据量不大时不做任何预采样
    n_skus = raw_df[sku_col].nunique() if sku_col else 0
    if len(raw_df) <= MAX_ROWS and n_skus <= MAX_SKUS:
        return raw_df, None
    original_rows = len(raw_df)
    notes = []
    df = raw_df
    # 步骤1: 时间截取 —— 只保留最近90天
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        max_date = dates.max()
        cutoff = max_date - pd.Timedelta(days=MAX_DAYS)
        mask = dates > cutoff
        df = df[mask].copy()
        notes.append(f"最近{MAX_DAYS}天")
    # 步骤2: 商品截取 —— 商品数仍太多时,按总销量降序只保留前200个
    if sku_col and df[sku_col].nunique() > MAX_SKUS:
        if sales_col:
            sales_num = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)
            sku_sales = pd.DataFrame({"_sku": df[sku_col].values, "_sales": sales_num.values})
            top_skus = sku_sales.groupby("_sku")["_sales"].sum().nlargest(MAX_SKUS).index
        else:
            top_skus = df[sku_col].value_counts().nlargest(MAX_SKUS).index
        df = df[df[sku_col].isin(top_skus)].copy()
        notes.append(f"销量前{MAX_SKUS}的商品")
    # 步骤3: 门店截取 —— 时间和商品截取后行数仍超标时,按销量只保留头部门店
    if store_col and len(df) > MAX_ROWS:
        n_stores = df[store_col].nunique()
        if n_stores > MAX_STORES:
            if sales_col:
                sales_num = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)
                store_sales = pd.DataFrame({"_store": df[store_col].values, "_sales": sales_num.values})
                top_stores = store_sales.groupby("_store")["_sales"].sum().nlargest(MAX_STORES).index
            else:
                top_stores = df[store_col].value_counts().nlargest(MAX_STORES).index
            df = df[df[store_col].isin(top_stores)].copy()
            notes.append(f"销量前{MAX_STORES}家门店")
    if not notes:
        return df, None
    print(f"[预采样] {original_rows}行 -> {len(df)}行(标准化前截取), 保留:{'、'.join(notes)}")
    note = (f"您的数据有{original_rows:,}行，已自动保留{'、'.join(notes)}进行分析，"
            f"结果依然有参考价值")
    return df, note


def _auto_sample(df: pd.DataFrame):
    """超大文件自动采样:行数或商品数太多时,只保留最近90天、销量前200的商品

    避免后台逐商品计算(缺货恢复/因果推断/补货/预测)在超大数据上卡死。
    返回 (采样后的df, 面向商家的采样说明文字或None);小数据集原样返回,完全不受影响。
    """
    MAX_ROWS = 1_000_000
    MAX_SKUS = 200
    MAX_DAYS = 90
    MAX_STORES = 20
    n_skus = df["sku_id"].nunique() if "sku_id" in df.columns else 0
    if len(df) <= MAX_ROWS and n_skus <= MAX_SKUS:
        return df, None
    original_rows = len(df)
    notes = []
    # 步骤1: 时间截取 —— 每个商品只保留最近90天的数据
    if "date" in df.columns:
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=MAX_DAYS)
        df = df[df["date"] > cutoff].copy()
        notes.append(f"最近{MAX_DAYS}天")
    # 步骤2: 商品截取 —— 商品数仍太多时,按总销量降序只保留前200个
    if "sku_id" in df.columns and "sales" in df.columns:
        n_skus = df["sku_id"].nunique()
        if n_skus > MAX_SKUS:
            top_skus = df.groupby("sku_id")["sales"].sum().nlargest(MAX_SKUS).index
            df = df[df["sku_id"].isin(top_skus)].copy()
            notes.append(f"销量前{MAX_SKUS}的商品")
    # 步骤3: 门店截取 —— 时间和商品截取后行数仍超标时,按销量只保留头部门店
    if "store_id" in df.columns and len(df) > MAX_ROWS:
        n_stores = df["store_id"].nunique()
        if n_stores > MAX_STORES:
            if "sales" in df.columns:
                top_stores = df.groupby("store_id")["sales"].sum().nlargest(MAX_STORES).index
            else:
                top_stores = df["store_id"].value_counts().nlargest(MAX_STORES).index
            df = df[df["store_id"].isin(top_stores)].copy()
            notes.append(f"销量前{MAX_STORES}家门店")
    # 缺少日期/商品列时无从截取,交给后续必填字段校验去提示商家
    if not notes:
        return df, None
    print(f"[自动采样] {original_rows}行 -> {len(df)}行, 保留:{'、'.join(notes)}")
    note = (f"您的数据有{original_rows:,}行，已自动保留{'、'.join(notes)}进行分析，"
            f"结果依然有参考价值")
    return df, note


def _save_upload_history(final_adapter: DataAdapter, file_names: list):
    """上传成功后保存数据副本与元信息,方便商户下次一键重新加载

    保存失败只打印警告,不影响本次上传(历史记录是附加功能,不能挡住正常使用)。
    """
    try:
        uploads = _uploads_dir()
        uploads.mkdir(parents=True, exist_ok=True)
        # 展示名:单文件用文件名,多文件用顿号连接
        display_name = file_names[0] if len(file_names) == 1 else "、".join(file_names)
        # 同名文件去重:同一文件重复上传时,先清理旧副本,历史里只保留最新一条
        for old_meta_file in uploads.glob("*.meta.json"):
            try:
                old_meta = json.loads(old_meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue  # 旧元信息损坏跳过,不挡新记录保存
            if old_meta.get("filename") == display_name:
                old_parquet = uploads / old_meta.get("file_path", "")
                old_parquet.unlink(missing_ok=True)
                old_meta_file.unlink(missing_ok=True)
                print(f"[上传历史] 已清理同名旧副本: {old_meta_file.name}")
        # 文件名安全化:去掉Windows文件名不允许的字符
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", display_name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{ts}_{safe_name}"
        parquet_path = uploads / f"{base}.parquet"
        meta_path = uploads / f"{base}.meta.json"
        # 保存原始数据副本(保留商户原始列名,重新加载时可再次走完整的识别与清洗流程)
        raw = final_adapter.raw_df
        try:
            raw.to_parquet(parquet_path, index=False)
        except (TypeError, ValueError):
            # 个别列类型混杂(既有文字又有数字)导致存不了时,统一转成文字再存
            safe_df = raw.copy()
            for col in safe_df.columns:
                if safe_df[col].dtype == object:
                    safe_df[col] = safe_df[col].astype(str)
            safe_df.to_parquet(parquet_path, index=False)
        # 保存元信息
        meta = {
            "filename": display_name,
            "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            "rows": int(len(raw)),
            "cols": int(len(raw.columns)),
            "tier": final_adapter.tier,
            "file_path": parquet_path.name,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[上传历史] 已保存副本: {parquet_path.name}")
    except Exception as e:
        print(f"[上传历史] 保存失败(不影响本次上传): {e}")


@router.get("/builtin")
async def list_builtin_datasets():
    """列出可用的内置示例数据集"""
    session = DataSession.get()
    return {"datasets": DataAdapter.list_builtin(session.datasets_dir)}


@router.post("/builtin/{dataset_id}")
async def load_builtin_dataset(dataset_id: str):
    """加载指定的内置示例数据集"""
    session = DataSession.get()
    try:
        adapter = DataAdapter.load_builtin(session.datasets_dir, dataset_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    # 标记为内置数据集
    adapter.profile["data_source_type"] = "builtin"
    session.set_adapter(adapter)
    # 刷新分析服务缓存
    AnalysisService.get()._invalidate_cache()
    # 后台预热全量计算缓存,用户首次请求秒级返回
    AnalysisService.get().preload_async()
    # 分层数据验证(非阻断,只提示)
    validation = validate_data(adapter.df)
    return {
        "message": f"已加载数据集: {adapter.source_name}",
        "tier": adapter.tier,
        "tier_label": {"basic": "基础版", "advanced": "进阶版", "full": "完整版"}[adapter.tier],
        "profile": adapter.profile,
        "validation": validation,
    }


@router.post("/upload")
async def upload_data_files(files: list[UploadFile] = File(...)):
    """上传一个或多个 CSV / Excel / Parquet / JSON 文件,多文件自动按日期+商品编号合并,缺失字段留空"""
    if not files:
        raise HTTPException(status_code=400, detail="未提供文件")
    adapters = []
    file_names = []
    pre_sample_notes = []
    for file in files:
        if not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower()
        if suffix not in (".csv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"):
            raise HTTPException(status_code=400, detail=f"文件「{file.filename}」格式不支持,仅支持 CSV / Excel / Parquet / JSON")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            adapter = DataAdapter().load_file(tmp_path)
            # 超大文件预采样:在昂贵的标准化之前先截取,避免对几百万行做完整清洗
            raw_sampled, pre_note = _pre_sample_raw(adapter.raw_df)
            if pre_note:
                adapter.raw_df = raw_sampled
                pre_sample_notes.append(pre_note)
            adapter = adapter.standardize()
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"文件「{file.filename}」解析失败: {str(e)}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        if len(adapter.df) == 0:
            raise HTTPException(status_code=422, detail=f"文件「{file.filename}」解析后无有效数据行,请检查格式")
        adapters.append(adapter)
        file_names.append(file.filename)
    if not adapters:
        raise HTTPException(status_code=400, detail="未提供有效文件")
    # 单文件直接用,多文件自动合并
    try:
        final_adapter = adapters[0] if len(adapters) == 1 else DataAdapter.merge_adapters(adapters)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    # 超大文件自动采样:数据太多会导致后台逐商品计算卡死,先截取到可分析的范围
    sampled_df, sample_note = _auto_sample(final_adapter.df)
    if sample_note:
        # 采样后重新走完整标准化流程(字段识别/清洗/分档/画像全部刷新)
        final_adapter = DataAdapter().load_dataframe(
            sampled_df, name=final_adapter.source_name
        ).standardize()
    # 校验必填字段:缺了分析没法做,直接告诉商家缺什么
    final_cols = set(final_adapter.df.columns)
    missing = []
    if "sales" not in final_cols:
        missing.append("销量(列名类似:销量、销售数量、Sales、Quantity)")
    if "date" not in final_cols:
        missing.append("日期(列名类似:日期、Date、销售日期)")
    if "sku_id" not in final_cols:
        missing.append("商品编号(列名类似:商品编码、SKU、货号、Item)")
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"您的数据里缺少这些必需的信息:{'、'.join(missing)}。请检查表格里有没有对应的列,这是做分析的前提。"
        )
    # 分层数据验证(非阻断,只提示问题不挡路)
    validation = validate_data(final_adapter.df)
    # 标记为用户上传的自有数据
    final_adapter.profile["data_source_type"] = "custom"
    session = DataSession.get()
    session.set_adapter(final_adapter)
    AnalysisService.get()._invalidate_cache()
    # 后台预热全量计算缓存,用户首次请求秒级返回
    AnalysisService.get().preload_async()
    # 保存上传历史副本(失败不阻断上传,商户仍可正常使用刚上传的数据)
    _save_upload_history(final_adapter, file_names)
    if len(file_names) == 1:
        msg = f"数据上传成功: {file_names[0]}"
    else:
        msg = f"已合并 {len(file_names)} 个文件: {'、'.join(file_names)}"
    final_note = pre_sample_notes[0] if pre_sample_notes else sample_note
    if final_note:
        msg = f"{msg}。{final_note}"
    return {
        "message": msg,
        "tier": final_adapter.tier,
        "tier_label": _tier_label(final_adapter.tier),
        "profile": final_adapter.profile,
        "validation": validation,
    }


@router.get("/suggest-supplement")
async def suggest_supplement():
    """分析当前数据缺什么关键信息,给商户列出补充上传建议(通俗语言)"""
    _require_loaded()
    session = DataSession.get()
    try:
        return session.adapter.get_supplement_suggestions()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/supplement")
async def supplement_data(files: list[UploadFile] = File(...)):
    """补充上传数据:自动按日期+商品编号合并到现有数据,只补空位不覆盖已有数据"""
    _require_loaded()
    if not files:
        raise HTTPException(status_code=400, detail="未提供文件")
    supp_frames = []
    file_names = []
    for file in files:
        if not file.filename:
            continue
        suffix = Path(file.filename).suffix.lower()
        if suffix not in (".csv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"):
            raise HTTPException(status_code=400, detail=f"文件「{file.filename}」格式不支持,仅支持 CSV / Excel / Parquet / JSON")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        try:
            if suffix in (".xlsx", ".xls"):
                supp_df = pd.read_excel(tmp_path)
            elif suffix == ".parquet":
                supp_df = pd.read_parquet(tmp_path)
            elif suffix == ".json":
                supp_df = pd.read_json(tmp_path)
            elif suffix == ".jsonl":
                supp_df = pd.read_json(tmp_path, lines=True)
            else:
                supp_df = pd.read_csv(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"文件「{file.filename}」解析失败: {str(e)}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        supp_frames.append(supp_df)
        file_names.append(file.filename)
    if not supp_frames:
        raise HTTPException(status_code=400, detail="未提供有效文件")
    # 多个补充文件先拼到一起,合并时同一日期+商品会自动去重
    supp_raw = pd.concat(supp_frames, ignore_index=True) if len(supp_frames) > 1 else supp_frames[0]

    session = DataSession.get()
    adapter = session.adapter
    supp_name = "、".join(file_names)
    try:
        stats = adapter.merge_supplement(supp_raw, supp_name=supp_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 刷新分析缓存,后台预热
    svc = AnalysisService.get()
    svc._invalidate_cache()
    svc.preload_async()

    # 保存上传历史(文件名标注"补充",方便商户识别)
    _save_upload_history(adapter, [f"{name}(补充)" for name in file_names])

    # 分层数据验证(非阻断,只提示)
    validation = validate_data(adapter.df)

    # 用通俗语言描述合并结果
    if stats["merged_fields"]:
        added_labels = "、".join(_field_label(f) for f in stats["merged_fields"])
        msg = f"已把您补充的{added_labels}数据合并进来，补上了 {stats['filled_rows']} 行记录"
    else:
        msg = "补充的数据现有表格里已经有了，不需要再补"
    return {
        "message": msg,
        "tier": adapter.tier,
        "tier_label": _tier_label(adapter.tier),
        "profile": adapter.profile,
        "validation": validation,
        "supplement_stats": stats,
    }


@router.get("/history")
async def list_upload_history():
    """列出所有上传历史记录,按上传时间倒序(最新的排在最前面)"""
    uploads = _uploads_dir()
    items = []
    if uploads.exists():
        for meta_file in uploads.glob("*.meta.json"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue  # 元信息损坏的记录跳过,不影响其他记录展示
            tier = meta.get("tier", "basic")
            items.append({
                "id": meta_file.name.removesuffix(".json"),
                "filename": meta.get("filename", ""),
                "uploaded_at": meta.get("uploaded_at", ""),
                "rows": meta.get("rows", 0),
                "cols": meta.get("cols", 0),
                "tier": tier,
                "tier_label": _tier_label(tier),
            })
    items.sort(key=lambda x: x["uploaded_at"], reverse=True)
    # 同名文件只保留最新一条(双保险:保存时已清理旧副本,此处再按文件名去重)
    seen_filenames = set()
    deduped = []
    for it in items:
        fn = it["filename"]
        if fn in seen_filenames:
            continue
        seen_filenames.add(fn)
        deduped.append(it)
    return {"items": deduped}


@router.post("/history/{record_id}/reload")
async def reload_history(record_id: str):
    """一键重新加载某次上传的数据(走完整的字段识别与数据清洗流程)"""
    uploads = _uploads_dir()
    meta_path = uploads / f"{record_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="找不到这条上传记录,可能已被删除")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记录信息读取失败: {e}")
    parquet_path = uploads / meta.get("file_path", "")
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="找不到对应的数据文件,可能已被手动删除")
    display_name = meta.get("filename", "历史数据")
    try:
        df = pd.read_parquet(parquet_path)
        # 预采样:标准化之前先截取超大历史数据
        df, pre_note = _pre_sample_raw(df)
        adapter = DataAdapter().load_dataframe(df, name=display_name).standardize()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"数据重新加载失败: {e}")
    if len(adapter.df) == 0:
        raise HTTPException(status_code=422, detail="这条记录的数据解析后无有效数据行")
    # 合并后仍超大的安全网(预采样后通常不会触发)
    sampled_df, sample_note = _auto_sample(adapter.df)
    if sample_note:
        adapter = DataAdapter().load_dataframe(sampled_df, name=display_name).standardize()
    if pre_note and not sample_note:
        sample_note = pre_note
    # 标记为用户上传的自有数据
    adapter.profile["data_source_type"] = "custom"
    session = DataSession.get()
    session.set_adapter(adapter)
    AnalysisService.get()._invalidate_cache()
    # 后台预热全量计算缓存,用户首次请求秒级返回
    AnalysisService.get().preload_async()
    # 分层数据验证(非阻断,只提示问题不挡路)
    validation = validate_data(adapter.df)
    reload_msg = f"已重新加载数据: {display_name}"
    if sample_note:
        reload_msg = f"{reload_msg}。{sample_note}"
    return {
        "message": reload_msg,
        "tier": adapter.tier,
        "tier_label": _tier_label(adapter.tier),
        "profile": adapter.profile,
        "validation": validation,
    }


@router.delete("/history/{record_id}")
async def delete_history(record_id: str):
    """删除一条上传记录(同时删除数据副本)"""
    uploads = _uploads_dir()
    meta_path = uploads / f"{record_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="找不到这条上传记录,可能已被删除")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    # 删除数据副本(文件不在了也不报错)
    parquet_path = uploads / meta.get("file_path", f"{record_id.removesuffix('.meta')}.parquet")
    parquet_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
    return {"message": "已删除"}


@router.get("/profile")
async def get_data_profile():
    """获取当前数据集的质量画像"""
    _require_loaded()
    session = DataSession.get()
    return session.adapter.profile


@router.get("/preload-status")
async def get_preload_status():
    """查询后台分析预热进度,供前端展示进度条。

    无需认证,无需数据已加载:未加载数据时返回 done=True,前端不显示进度条。
    """
    return AnalysisService.get().get_preload_status()


@router.get("/status")
async def get_data_status():
    """查询当前数据加载状态"""
    session = DataSession.get()
    loaded = session.is_loaded()
    result = {"loaded": loaded}
    if loaded:
        result["tier"] = session.get_tier()
        result["tier_label"] = {"basic": "基础版", "advanced": "进阶版", "full": "完整版"}[session.get_tier()]
        result["source"] = session.adapter.source_name
        result["rows"] = len(session.adapter.df)
        result["n_skus"] = int(session.adapter.df["sku_id"].nunique()) if "sku_id" in session.adapter.df.columns else 0
        result["data_source_type"] = session.adapter.profile.get("data_source_type", "builtin")
    return result


@router.get("/preview")
async def preview_data(rows: int = 20):
    """预览当前数据集的前N行"""
    _require_loaded()
    svc = AnalysisService.get()
    df = svc.get_df()
    n = min(rows, len(df))
    records = df.head(n).copy()
    # 日期列转字符串
    for col in records.columns:
        if str(records[col].dtype).startswith("datetime"):
            records[col] = records[col].dt.strftime("%Y-%m-%d")
    # 用 to_json 确保所有类型(含numpy类型、NaN)都能正确序列化,避免500错误
    import json as _json
    safe_rows = _json.loads(records.to_json(orient="records", force_ascii=False))
    return {"columns": list(df.columns), "rows": safe_rows, "total": len(df)}


@router.post("/enrich")
async def enrich_local_factors(body: dict):
    """
    智能体自动补齐当地影响因素:
    输入门店所在城市名,自动获取该城市的历史天气(气温/降水)和法定节假日标记,
    合并到当前数据集中参与因果分析。无需API密钥。
    """
    _require_loaded()
    city = body.get("city", "").strip()
    if not city:
        raise HTTPException(status_code=422, detail="请输入门店所在城市,如:北京、上海、广州")
    from backend.core.external_factors import enrich_factors
    svc = AnalysisService.get()
    df = svc.get_df()
    result = enrich_factors(df, city)
    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("message", "补齐失败"))
    # 更新数据会话中的DataFrame
    enriched_df = result.pop("df")
    session = DataSession.get()
    session.adapter.df = enriched_df
    # 重新标准化分档与画像
    session.adapter._detect_tier()
    session.adapter._build_profile()
    svc._invalidate_cache()
    # 后台预热全量计算缓存,用户首次请求秒级返回
    svc.preload_async()
    return result


@router.post("/remap-column")
async def remap_column(body: dict):
    """
    商户手动指定未识别列的含义:
    输入原始列名和标准字段名,系统重新标准化数据集并刷新分析缓存。
    field传空字符串表示将该列标记为"不使用"。
    """
    _require_loaded()
    column = (body.get("column") or "").strip()
    field = (body.get("field") or "").strip() or None
    if not column:
        raise HTTPException(status_code=422, detail="缺少列名")
    session = DataSession.get()
    adapter = session.adapter
    try:
        adapter.remap_column(column, field)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    svc = AnalysisService.get()
    svc._invalidate_cache()
    # 后台预热全量计算缓存,用户首次请求秒级返回
    svc.preload_async()
    field_labels = {"date":"日期","store_id":"门店编号","sku_id":"商品编号","sku_name":"商品名称","category":"商品分类","sales":"销量","demand":"需求","price":"售价","base_price":"标准售价","is_promo":"是否促销","discount":"折扣","temperature":"温度","precipitation":"降水量","is_weekend":"是否周末","is_holiday":"是否节假日","holiday_name":"节假日名称","inventory":"库存","in_transit":"在途库存","stockout":"是否缺货","waste_qty":"损耗数量","near_expiry":"临期数量","near_expiry_price":"临期售价","cost":"成本","shelf_life":"保质期","lead_time":"补货周期","unit":"单位","gross_margin":"毛利","stockout_loss":"缺货损失"}
    label = field_labels.get(field, field) if field else "不使用"
    return {"message": f"已将列「{column}」识别为: {label}", "profile": adapter.profile}