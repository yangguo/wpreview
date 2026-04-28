# 底稿复核工具（NLP/规则）

通过 `openpyxl` 直接读取 Excel 底稿结构（不做图像识别），并基于规则/启发式（可扩展到 LLM）对审计底稿的方法与一致性进行复核。

## 架构

1. 文档预处理（结构化抽取）
- 通过 `openpyxl` 读取单元格坐标、行列位置、合并单元格范围
- 识别连续的表格区域
- 将字段映射为统一 Schema：
  - `control_id`
  - `control_description`
  - `audit_objective`
  - `audit_procedure`
  - `sample_selection_method`
  - `sample_size`
  - `test_steps`
  - `test_result`
  - `exception_flag`
  - `conclusion`

2. 复核层（规则/可扩展 LLM）
- 覆盖性：执行步骤是否覆盖标准程序/检查要点
- 方法性：证据充分性、样本总体/抽样基准、样本量、例外闭环、结论与证据一致性
- 逻辑一致性：同一底稿内是否存在矛盾
- 跨字段一致性：控制/程序/样本/结果/结论/例外/期间等是否一致

3. 报告生成
- 输出问题清单（定位、原文摘录、判定依据、整改建议）
- 可扩展：保存结构化抽取结果与文档报告

## 安装

```bash
pip install -r requirements.txt
```

创建 `.env` 文件：

```bash
OPENAI_API_KEY=your_openai_api_key
```

## 使用

```bash
python excel_image_review.py path/to/file.xlsx
python excel_image_review.py path/to/file.xlsx -o output_dir -m gpt-4o
```

参数：

```text
  excel_file        Excel文件路径
  -o, --output DIR  输出目录（默认：output）
  -m, --model MODEL 使用的模型（默认：gpt-4o）
```

输出：
- `output/<sheet_name>_structured.json` (structured extraction payload)
- `output/review_report.docx` (review report)

使用示例数据（脚本会生成一个示例 Excel）：

```bash
python create_sample_excel.py
python excel_image_review.py path/to/sample_data.xlsx
```

以代码方式调用：

```python
from excel_image_review import ExcelImageReviewer

reviewer = ExcelImageReviewer("data.xlsx", output_dir="results")
reviewer.process_excel()
report_path = reviewer.generate_report()
```

## analyze_excel.py 使用说明

`analyze_excel.py` 用于对 ITGC 底稿（Excel）进行复核，并输出复核报告（支持 `.xlsx` 或 `.txt`）。

当前版本默认会调用 LLM（需配置 `.env`），主要用于：
- 基于“检查要点.xlsx”对指定 Sheet 做要点复核（存在问题/不确定则入“问题清单”）
- 对“标准审计程序 vs 执行审计程序”做对应性检查（汇总与问题明细在“LLM对应性”页签）
- 对“问题清单”进行二次复核，补充 LLM 结论字段

### 基本用法

```bash
python analyze_excel.py -i "path\to\底稿.xlsx" -o "path\to\复核报告.xlsx"
```

### LLM 配置

在 `analyze_excel.py` 同目录创建 `.env`（或设置同名环境变量），至少需要 API Key：

```bash
OPENAI_API_KEY=your_key
```

可选配置（如使用兼容 OpenAI 的第三方接口）：

```bash
OPENAI_BASE_URL=https://your-host/v1
OPENAI_MODEL=gpt-4o
```

### 参数说明

```text
  -i, --input            待复核的底稿Excel路径（.xlsx）
  -o, --output           输出报告路径（.xlsx 或 .txt）
  -k, --checkpoints      检查要点Excel路径（.xlsx，可选）
  -s, --sheets           指定只检查的Sheet（控制点页签），逗号/空格分隔（可选）
```

### 输出说明（Excel 报告）

当 `--output` 以 `.xlsx` 结尾时，输出报告包含以下页签：
- `汇总`：基本信息（输入文件/检查要点/本次检查范围）与统计汇总
- `问题清单`：逐条问题列表（含 Sheet/单元格定位、原文摘录、判定依据、整改建议）
- `LLM对应性`：对应性检查统计 + 问题明细（不包含“全量结果”明细；无冻结窗格）

说明：
- “关键角色候选”已合并进 `汇总` 页签，不再单独输出页签。
- `-k/--checkpoints` 不传时，不做“检查要点”复核；其余复核仍会执行。

### 示例

1) 全量检查（输出 Excel 报告）

```bash
python analyze_excel.py -i "path\to\workpaper.xlsx" -o "path\to\excel_analysis_report.xlsx"
```

2) 带检查要点（基于 `检查要点.xlsx` 对对应 Sheet 做 LLM 要点复核）

```bash
python analyze_excel.py -i "path\to\workpaper.xlsx" -o "path\to\excel_analysis_report_with_checkpoints.xlsx" -k "path\to\检查要点.xlsx"
```

3) 只检查指定控制点页签（例如 SA-4c、SA-5）

```bash
python analyze_excel.py -i "path\to\workpaper.xlsx" -o "path\to\excel_analysis_report_SA4c_SA5.xlsx" -k "path\to\检查要点.xlsx" -s "SA-4c,SA-5"
```

## 项目结构

```text
├── excel_image_review.py                  # 主脚本（结构化审阅）
├── create_sample_excel.py                 # 示例数据生成
├── test_excel_review.py                   # 端到端（mock）测试
├── test_excel_image_review_libreoffice.py # 结构/schema 单元测试
└── requirements.txt
```

## 许可

许可信息以仓库内文件为准。
