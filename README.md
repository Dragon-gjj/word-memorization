# 百词斩 — 单词背诵小工具

这是一个基于 Streamlit 的简易单词背诵工具，功能包括：导入单词表、错题本、记忆模式、考察模式（拼写 / 选择题）以及 AI 助记聊天栏（支持 OpenAI 与国内兼容接口 千问 qianwen）。

## 主要文件

- `百词斩.py` — 主应用（Streamlit）。
- `requirements.txt` — 依赖清单。
- `sample_words.csv` — 示例单词表（可选）。
- `wrongbook.json` — 错题本（运行后生成并保存在同目录）。

## 快速开始（Windows / PowerShell）

建议在虚拟环境中安装依赖并运行：

```powershell
python -m venv .venv;
.\.venv\Scripts\Activate.ps1;
python -m pip install --upgrade pip setuptools wheel;
python -m pip install -r requirements.txt
```

启动应用：

```powershell
streamlit run 百词斩.py
```

## AI 助记（OpenAI / 千问 qianwen）

应用支持两类 AI 后端：

- OpenAI（推荐使用环境变量 `OPENAI_API_KEY` 或在左侧面板输入并保存）。
- 千问（qianwen）兼容端点：适用于国内或厂商提供的 OpenAI-compatible 接口。需在左侧面板填写 API Base（例如：`https://openapi.qianwen.aliyun.com/v1/chat/completions`）与 API Key 并保存。

注意事项：

- 在应用左侧“AI 服务配置”可选择提供方（`openai` / `qianwen`），输入 API Base 与 API Key 并点击“保存配置”。配置仅保存在当前运行的环境（进程）中。
- 若未配置或网络/密钥不可用，应用会回退到本地模拟回答（仅用于简单演示）。
- 不同服务的密钥放置方式可能不同（部分服务要求 `Authorization: Bearer <key>`，有些要求 `X-API-KEY`），当前默认使用常见的 Bearer 方式。如你的服务要求不同，请在提供方文档中确认并在代码中调整 `call_qianwen` 的请求头。

## 导入词表格式

- 支持 CSV（含表头 `english` 和 `chinese`）或纯文本，每行格式 `english<TAB>chinese` / `english,chinese`。
- 导入后会在左侧显示解析预览（前 10 条），需点击“确认导入”后才会把词表写入应用会话状态。

## 应用模式说明

- 记忆模式（记忆模式）：逐词浏览，支持“上一个 / 下一个 / 标记为不会（加入错题本）”。
- 错题记忆模式（错题记忆模式）：专门复习已加入的错题本，可逐条浏览并从错题本移除已掌握项。
- 考察模式（考察模式）：支持两种考察方式：拼写检查（输入英文）与选择题（单选）。可从“全部单词”、“错题本”或“从记忆位置往前抽查”中选择题库并随机抽题。考察支持提交答案、即时反馈、错题加入与“学会了/结束考察”等操作。

考察结束后会显示本次成绩（正确数 / 总数）并有鼓励语与气球动画。

## 错题本（wrongbook.json）

- 错题本文件 `wrongbook.json` 会保存在应用脚本同目录下。应用内提供查看、手动清空、以及在答题环节自动加入/移除错题的功能。

## 依赖与兼容性

- 推荐 Python 版本：3.8 - 3.11（部分平台/版本可能会导致依赖（如 numpy）从源码编译，进而出现构建错误）。如果遇到 pip 安装 numpy 或其它依赖失败的问题，建议使用 Python 3.11 或通过 conda 安装预编译包。
- 依赖已在 `requirements.txt` 中列出，请使用虚拟环境安装。

## 常见问题与排错

- 如果 AI 功能提示“未配置 OpenAI API Key”或“未配置千问 API 地址”，请在左侧面板填写并保存对应的 Key 与 Base，或设置环境变量：

```powershell
$env:OPENAI_API_KEY = "sk-..."
# 或者在当前 PowerShell 会话中临时设置千问配置：
$env:AI_PROVIDER = "qianwen";
$env:AI_API_BASE = "https://openapi.qianwen.aliyun.com/v1/chat/completions";
$env:AI_API_KEY = "your_qianwen_key"
```

- 如果导入后没有单词，请检查 CSV 是否包含 `english` 和 `chinese` 两列，或确保每行以 tab/comma 分隔两个字段。

- 如果在安装依赖时遇到编译或二进制相关错误，优先尝试：升级 pip/setuptools/wheel；或切换到 Python 3.11；或使用 conda 创建环境并安装 numpy 等二进制包。



