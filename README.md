# 百词斩 — 单词背诵小工具

这是一个基于 Streamlit 的简易单词背诵工具，包含：导入单词表、错题本、记忆模式、考察模式和 AI 助记聊天栏（可连接 OpenAI）。

主要文件

- `百词斩.py` — 主应用（使用 Streamlit）。
- `requirements.txt` — 依赖清单。
- `sample_words.csv` — 示例单词表。
- `wrongbook.json` — 错题本（运行后生成）。

快速开始

1. 建议在虚拟环境中安装依赖：

```powershell
python -m venv .venv;
.\.venv\Scripts\Activate.ps1;
python -m pip install -r requirements.txt
```

2. 运行应用：

```powershell
streamlit run 百词斩.py
```

3. 若想使用 AI 聊天功能：

- 设置环境变量 `OPENAI_API_KEY` 为你的 API Key（示例）：

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

- 若未配置 Key，应用会回退到本地模拟回答。

注意

- 导入的 CSV 文件应包含两列 `english` 和 `chinese` 或每行 `english<TAB>chinese`。
- 错题本会保存在 `wrongbook.json` 中。

如需修改或添加功能，请反馈具体需求。