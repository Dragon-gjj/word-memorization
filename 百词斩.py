import streamlit as st
import pandas as pd
import json
import io
import os
import random
from typing import List, Dict

try:
	import openai
except Exception:
	openai = None

try:
	import requests
except Exception:
	requests = None

# 常量
SYSTEM_PROMPT = (
	"你是一个只回答关于单词、词义、拼写、例句以及记忆方法的助教，不回答其他内容。"
	"若用户问与单词无关的问题，则简短拒绝并引导回到单词主题。用中文回答。"
)
ALLOWED_KEYWORDS = [
	"单词",
	"记忆",
	"拼写",
	"词义",
	"意思",
	"例句",
	"近义",
	"联想",
	"记法",
	"词根",
	"词缀",
	"如何记",
	"怎么记",
]

# 文件路径（在当前脚本目录）
BASE_DIR = os.path.dirname(__file__)
WRONGBOOK_PATH = os.path.join(BASE_DIR, "wrongbook.json")
SAMPLE_CSV = os.path.join(BASE_DIR, "sample_words.csv")


def load_wrongbook() -> List[Dict]:
	if os.path.exists(WRONGBOOK_PATH):
		try:
			with open(WRONGBOOK_PATH, "r", encoding="utf-8") as f:
				return json.load(f)
		except Exception:
			return []
	return []


def save_wrongbook(wrong: List[Dict]):
	with open(WRONGBOOK_PATH, "w", encoding="utf-8") as f:
		json.dump(wrong, f, ensure_ascii=False, indent=2)


def parse_wordfile(file_bytes) -> List[Dict]:
	"""接受 CSV 或纯文本：每行 english[,\t]chinese 或 CSV 带表头 english,chinese"""
	try:
		s = file_bytes.decode("utf-8")
	except Exception:
		s = file_bytes.decode("gbk", errors="ignore")
	# 尝试 CSV
	if "\n" not in s:
		s = s.replace("\\r", "\\n")
	lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
	rows = []
	# 如果第一行包含英文列名则用 pandas
	first = lines[0]
	if "," in first or "\t" in first:
		sep = "," if first.count(",") >= first.count("\t") else "\t"
		try:
			df = pd.read_csv(io.StringIO(s), sep=sep)
			if "english" in df.columns and "chinese" in df.columns:
				for _, r in df.iterrows():
					rows.append({"english": str(r["english"]).strip(), "chinese": str(r["chinese"]).strip()})
				return rows
		except Exception:
			pass
	# 回退为简单解析：每行用 tab 或 comma 或 空格分割为两段
	for ln in lines:
		for sep in ["\t", ",", " "]:
			if sep in ln:
				parts = [p.strip() for p in ln.split(sep) if p.strip()]
				if len(parts) >= 2:
					rows.append({"english": parts[0], "chinese": parts[1]})
					break
		else:
			# 如果整行只有一个词，跳过
			continue
	return rows


def load_sample() -> List[Dict]:
	if os.path.exists(SAMPLE_CSV):
		return parse_wordfile(open(SAMPLE_CSV, "rb").read())
	return []


def call_qianwen(api_base: str, api_key: str, messages: List[Dict], user_input: str) -> str:
	"""向千问（qianwen）兼容端点发送请求并返回文本结果（若失败返回 None）。"""
	try:
		headers = {"Content-Type": "application/json"}
		if api_key:
			# 常见做法是放在 Authorization 或 X-API-KEY 中，请根据实际密钥要求调整
			headers["Authorization"] = f"Bearer {api_key}"
		payload = {
			"model": "qwen-turbo",
			"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages + [{"role": "user", "content": user_input}],
			"max_tokens": 400,
			"temperature": 0.6,
		}
		resp = requests.post(api_base, json=payload, headers=headers, timeout=15)
		if resp.status_code != 200:
			return f"千问 API 请求失败：{resp.status_code} {resp.text}"
		try:
			data = resp.json()
		except Exception:
			return resp.text[:1000]

		if isinstance(data, dict):
			# OpenAI 兼容
			if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
				return data["choices"][0].get("message", {}).get("content", "").strip()
			# 千问可能返回 data.content
			if "data" in data and isinstance(data["data"], dict) and "content" in data["data"]:
				return str(data["data"]["content"]).strip()
			if "answer" in data:
				return str(data["answer"]).strip()
		return str(data)[:1000]
	except Exception as e:
		return f"千问 API 调用出错：{e}"


def send_ai_query(messages: List[Dict], user_input: str) -> str:
	"""过滤用户输入，仅允许与单词记忆相关的问题；根据配置调用 qianwen 或 openai，失败时回退本地模拟回答。"""
	low = user_input.lower()
	if not any(k in low for k in ALLOWED_KEYWORDS) and len(low.split()) > 5:
		return "抱歉，我只能回答与单词和背记相关的问题。请把问题限定为单词的释义、拼写、例句或记忆方法等。"

	provider = os.environ.get("AI_PROVIDER", "openai")
	api_base = os.environ.get("AI_API_BASE", "")
	api_key = os.environ.get("AI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

	def local_reply():
		tokens = user_input.split()
		word = tokens[0] if tokens else "单词"
		return f"（本地模拟回答）关于 '{word}'：建议先记词义，再联想一个短句；例如：'{word}' — 例句：This is a sample sentence containing {word}."

	# qianwen 分支
	if provider == "qianwen":
		if not api_base:
			return "未配置千问 API 地址（AI_API_BASE）。请在左侧配置并保存。"
		if requests is None:
			return "未安装 requests，无法使用千问 API。请安装 requests 或选择 OpenAI。"
		return call_qianwen(api_base, api_key, messages, user_input) or local_reply()

	# openai 分支
	if openai is None:
		return local_reply()
	if not api_key:
		return "未配置 OpenAI API Key，无法使用 AI 功能。请在左侧配置。"
	try:
		chat = openai.ChatCompletion.create(
			model="gpt-3.5-turbo",
			messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages + [{"role": "user", "content": user_input}],
			temperature=0.6,
			max_tokens=400,
		)
		return chat.choices[0].message.content.strip()
	except Exception as e:
		return f"AI 请求失败：{e}"


def main():
	st.set_page_config(page_title="百词斩", layout="wide")
	st.title("百词斩 — 单词背诵小工具")

	# UI: 左-中-右 三栏
	left, mid, right = st.columns([1, 2, 1])

	# 会话状态初始化
	if "words" not in st.session_state:
		st.session_state.words = []  # list of dict {english, chinese}
	if "index" not in st.session_state:
		st.session_state.index = 0
	if "wrong_index" not in st.session_state:
		st.session_state.wrong_index = 0
	if "mode" not in st.session_state:
		st.session_state.mode = "memory"
	if "test_queue" not in st.session_state:
		st.session_state.test_queue = []
	if "test_progress" not in st.session_state:
		st.session_state.test_progress = {"asked": 0, "correct": 0, "total": 0}
	if "wrongbook" not in st.session_state:
		st.session_state.wrongbook = load_wrongbook()
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []
	# 保存记忆模式的当前位置，用于考察模式
	if "memory_position" not in st.session_state:
		st.session_state.memory_position = 0
	# 考察模式类型：拼写检查或选择题
	if "test_type" not in st.session_state:
		st.session_state.test_type = "spelling"  # "spelling" 或 "choice"
	# 存储选择题的选项（避免每次重新打乱）
	if "choice_options" not in st.session_state:
		st.session_state.choice_options = {}  # {题目索引: [选项列表]}
	# 存储当前题目的答题状态
	if "answer_submitted" not in st.session_state:
		st.session_state.answer_submitted = {}  # {题目索引: {"submitted": bool, "is_correct": bool}}

	with left:
		st.header("操作面板")
		uploaded = st.file_uploader("导入单词表（CSV: english,chinese 或 每行 english<TAB>chinese）", type=["csv", "txt"])
		if uploaded is not None:
			try:
				rows = parse_wordfile(uploaded.read())
				if rows:
					# 将解析结果放到待确认的会话状态，等待用户点击确认导入
					st.session_state.pending_import = rows
					st.info(f"已解析到 {len(rows)} 条词。请确认导入。")
					# 预览前 10 条
					try:
						preview_df = pd.DataFrame(rows).head(10)
						st.table(preview_df)
					except Exception:
						# 如果 DataFrame 构建失败，简单列出前几项
						for r in rows[:10]:
							st.write(r)
				else:
					st.warning("未解析到任何有效单词，请检查文件格式（english,chinese 或 english<TAB>chinese）。")
			except Exception as e:
				st.error(f"导入失败：{e}")

		# 如果有待确认的导入，显示确认/取消按钮
		if st.session_state.get("pending_import"):
			cols_imp = st.columns([1, 1])
			if cols_imp[0].button("确认导入"):
				st.session_state.words = st.session_state.pop("pending_import")
				st.session_state.index = 0
				st.success(f"已导入 {len(st.session_state.words)} 个单词")
			if cols_imp[1].button("取消导入"):
				st.session_state.pop("pending_import", None)
				st.info("已取消导入")

		if st.button("载入示例单词", key="load_sample"):
			sample = load_sample()
			if sample:
				st.session_state.words = sample
				st.session_state.index = 0
				st.success(f"已载入示例，共 {len(sample)} 个单词")
			else:
				st.warning("未找到示例文件 sample_words.csv")

		# --- AI 提供方与 API 配置（可选择国内自定义兼容接口）
		st.write("---")
		st.write("AI 服务配置（可选：OpenAI 或 千问 qianwen）")
		provider = st.selectbox("选择 AI 提供方", options=["openai", "qianwen"], index=0, key="ai_provider_select")
		# 默认从环境读取已有配置
		# 若选择千问（qianwen），给出常见的建议端点（可根据厂商文档调整）
		suggest_qianwen = "https://openapi.qianwen.aliyun.com/v1/chat/completions"
		default_base = os.environ.get("AI_API_BASE", "")
		if provider == "qianwen" and not default_base:
			default_base = suggest_qianwen
		default_key = os.environ.get("AI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
		base_input = st.text_input("自定义 API Base（例如：https://api.yourprovider.com/v1/chat）", value=default_base, key="ai_api_base_input")
		key_input = st.text_input("API Key", value=default_key, type="password", key="ai_api_key_input")
		cols_key = st.columns([1, 1])
		if cols_key[0].button("保存配置", key="save_config"):
			# 保存到会话环境（仅本次运行）
			os.environ["AI_PROVIDER"] = provider
			if base_input and base_input.strip():
				os.environ["AI_API_BASE"] = base_input.strip()
			else:
				if "AI_API_BASE" in os.environ:
					del os.environ["AI_API_BASE"]
			if key_input and key_input.strip():
				os.environ["AI_API_KEY"] = key_input.strip()
			else:
				if "AI_API_KEY" in os.environ:
					del os.environ["AI_API_KEY"]
			st.success("AI 配置已保存到当前会话（仅本次运行）")
		if cols_key[1].button("清除配置", key="clear_config"):
			for v in ("AI_PROVIDER", "AI_API_BASE", "AI_API_KEY"):
				if v in os.environ:
					del os.environ[v]
			st.success("已清除 AI 配置（仅本次运行）")
		# 显示当前检测到的状态
		curprov = os.environ.get("AI_PROVIDER", "openai")
		if curprov == "openai" and os.environ.get("OPENAI_API_KEY"):
			st.info("使用 OpenAI（检测到 OPENAI_API_KEY）。")
		elif curprov == "qianwen" and os.environ.get("AI_API_BASE"):
			st.info(f"使用千问 qianwen：{os.environ.get('AI_API_BASE')}")
		else:
			st.info("未检测到完整的 AI 配置，将使用本地模拟回答（如 OpenAI/千问 未配置）。")
		# 模式包括：记忆模式(memory)、错题记忆模式(wrong_memory)、考察模式(test)
		mode_map = {"记忆模式": 0, "错题记忆模式": 1, "考察模式": 2}
		default_idx = mode_map.get(st.session_state.mode, 0)
		st.radio("选择模式", options=["记忆模式", "错题记忆模式", "考察模式"], index=default_idx, key="mode_radio")
		st.session_state.mode = st.session_state.mode_radio

		st.write("错题本：")
		if st.button("查看/清空错题本", key="view_wrongbook"):
			st.session_state.wrongbook = load_wrongbook()
			if st.session_state.wrongbook:
				if st.button("清空错题本确认", key="clear_wrongbook"):
					st.session_state.wrongbook = []
					save_wrongbook([])
					st.success("已清空错题本")
			else:
				st.info("当前错题本为空")

		st.write(f"错题数量：{len(st.session_state.wrongbook)}")

	with mid:
		if st.session_state.mode == "记忆模式":
			st.header("记忆模式")
			if not st.session_state.words:
				st.info("请在左侧导入单词或载入示例")
			else:
				w = st.session_state.words[st.session_state.index % len(st.session_state.words)]
				st.subheader(w.get("english", ""))
				st.write(w.get("chinese", ""))
				
				# 使用callback函数处理按钮点击 - 确保从session_state获取当前状态
				def prev_word():
					print(f"[DEBUG] prev_word called - current index: {st.session_state.index}")
					st.session_state.index = (st.session_state.index - 1) % len(st.session_state.words)
					# 保存当前位置到memory_position
					st.session_state.memory_position = st.session_state.index
					print(f"[DEBUG] prev_word - new index: {st.session_state.index}")
				
				def next_word():
					print(f"[DEBUG] next_word called - current index: {st.session_state.index}")
					st.session_state.index = (st.session_state.index + 1) % len(st.session_state.words)
					# 保存当前位置到memory_position
					st.session_state.memory_position = st.session_state.index
					print(f"[DEBUG] next_word - new index: {st.session_state.index}")
				
				def mark_wrong():
					# 从session_state获取当前单词，避免闭包问题
					print(f"[DEBUG] mark_wrong called - index: {st.session_state.index}")
					print(f"[DEBUG] total words: {len(st.session_state.words)}")
					current_word = st.session_state.words[st.session_state.index % len(st.session_state.words)]
					print(f"[DEBUG] current_word: {current_word}")
					entry = {"english": current_word["english"], "chinese": current_word.get("chinese", "")}
					print(f"[DEBUG] entry to add: {entry}")
					if entry not in st.session_state.wrongbook:
						st.session_state.wrongbook.append(entry)
						save_wrongbook(st.session_state.wrongbook)
						print(f"[DEBUG] entry added to wrongbook")
					else:
						print(f"[DEBUG] entry already in wrongbook")
				
				cols = st.columns(3)
				cols[0].button("⏮️ 上一个", key="memory_prev", on_click=prev_word)
				cols[1].button("下一个", key="memory_next", on_click=next_word)
				cols[2].button("标记为不会（加入错题本）", key="memory_mark", on_click=mark_wrong)

		elif st.session_state.mode == "错题记忆模式":
			st.header("错题记忆模式")
			if not st.session_state.wrongbook:
				st.info("错题本为空，先在记忆/考察模式中加入错题")
			else:
				w = st.session_state.wrongbook[st.session_state.wrong_index % len(st.session_state.wrongbook)]
				st.subheader(w.get("english", ""))
				st.write(w.get("chinese", ""))
				cols = st.columns(3)
				if cols[0].button("⏮️ 上一个 错题", key="wrong_prev"):
					st.session_state.wrong_index = (st.session_state.wrong_index - 1) % len(st.session_state.wrongbook)
					st.rerun()
				if cols[1].button("下一个 错题", key="wrong_next"):
					st.session_state.wrong_index = (st.session_state.wrong_index + 1) % len(st.session_state.wrongbook)
					st.rerun()
				if cols[2].button("已掌握，移除错题本", key="wrong_remove"):
					entry = {"english": w["english"], "chinese": w.get("chinese", "")}
					st.session_state.wrongbook = [x for x in st.session_state.wrongbook if x != entry]
					save_wrongbook(st.session_state.wrongbook)
					st.success("已从错题本移除")

		else:
			st.header("考察模式")
			if not st.session_state.words and not st.session_state.wrongbook:
				st.info("请先导入单词或加入错题本")
			else:
				# 考察类型选择
				test_type = st.radio(
					"选择考察方式",
					options=["拼写检查", "选择题"],
					index=0 if st.session_state.test_type == "spelling" else 1,
					horizontal=True,
					key="test_type_radio"
				)
				st.session_state.test_type = "spelling" if test_type == "拼写检查" else "choice"
				
				test_source = st.selectbox("选择考察词库", options=["全部单词", "错题本", "从记忆位置往前抽查"])
				
				if test_source == "从记忆位置往前抽查":
					# 显示当前记忆位置信息
					if st.session_state.memory_position > 0:
						st.info(f"📍 记忆模式当前进度：已背诵到第 {st.session_state.memory_position + 1} 个单词，将从前 {st.session_state.memory_position} 个单词中抽查")
					else:
						st.info("📍 记忆模式刚开始，将从第一个单词开始抽查")
					
					total_available = st.session_state.memory_position if st.session_state.memory_position > 0 else len(st.session_state.words)
					n = st.number_input("考察数量", min_value=1, max_value=max(1, total_available), value=min(10, max(1, total_available)))
					
					if st.button("开始考察", key="start_test_from_memory"):
						if total_available == 0:
							st.warning("还没有背诵任何单词，请先在记忆模式中背诵一些单词")
						else:
							# 从memory_position之前取单词
							pool = st.session_state.words[:st.session_state.memory_position] if st.session_state.memory_position > 0 else st.session_state.words
							if pool:
								st.session_state.test_queue = random.sample(pool, min(n, len(pool)))
								st.session_state.test_progress = {"asked": 0, "correct": 0, "total": len(st.session_state.test_queue)}
								# 清空选择题选项缓存和答题状态
								st.session_state.choice_options = {}
								st.session_state.answer_submitted = {}
								st.success(f"已从前 {len(pool)} 个单词中随机抽取 {len(st.session_state.test_queue)} 个进行考察")
							else:
								st.warning("词库为空")
				else:
					total_available = len(st.session_state.words) if test_source == "全部单词" else len(st.session_state.wrongbook)
					n = st.number_input("考察数量", min_value=1, max_value=max(1, total_available), value=min(10, max(1, total_available)))
					if st.button("开始考察", key="start_test"):
						pool = st.session_state.words if test_source == "全部单词" else st.session_state.wrongbook
						if not pool:
							st.warning("所选词库为空")
						else:
							st.session_state.test_queue = random.sample(pool, min(n, len(pool)))
							st.session_state.test_progress = {"asked": 0, "correct": 0, "total": len(st.session_state.test_queue)}
							# 清空选择题选项缓存和答题状态
							st.session_state.choice_options = {}
							st.session_state.answer_submitted = {}

				if st.session_state.test_queue:
					qidx = st.session_state.test_progress["asked"]
					if qidx < st.session_state.test_progress["total"]:
						cur = st.session_state.test_queue[qidx]
						st.write(f"第 {qidx+1} / {st.session_state.test_progress['total']} 题")
						
						# 根据考察类型显示不同的题目
						if st.session_state.test_type == "spelling":
							# 拼写检查模式（改为先提交保留结果，再显示反馈）
							st.markdown(f"**中文提示：** {cur.get('chinese','')} ")
							ans = st.text_input("请输入英文拼写（不区分大小写）", key=f"ans_{qidx}")

							# 添加上一题和下一题按钮（浏览功能）
							nav_cols = st.columns([1, 1, 2])
							if nav_cols[0].button("⏮️ 上一题", key=f"nav_prev_{qidx}", disabled=(qidx == 0)):
								st.session_state.test_progress["asked"] = max(0, qidx - 1)
								st.rerun()
							if nav_cols[1].button("跳过（加入错题）", key=f"skip_{qidx}"):
								# 将当前单词加入错题本（若尚未存在），然后跳到下一题
								entry = {"english": cur.get('english'), "chinese": cur.get('chinese', '')}
								if entry not in st.session_state.wrongbook:
									st.session_state.wrongbook.append(entry)
									save_wrongbook(st.session_state.wrongbook)
									st.info("已加入错题本，已跳过当前题目")
								if qidx < st.session_state.test_progress["total"] - 1:
									st.session_state.test_progress["asked"] = qidx + 1
									st.rerun()
								else:
									st.info("已经是最后一题，已加入错题；可点击结束考察")

							submitted = st.session_state.answer_submitted.get(qidx, {}).get("submitted", False)
							is_correct = st.session_state.answer_submitted.get(qidx, {}).get("is_correct", False)

							if not submitted:
								if st.button("✅ 提交答案", key=f"submit_{qidx}"):
									user = ans.strip().lower()
									target = cur.get("english", "").strip().lower()
									if user == target:
										st.session_state.test_progress["correct"] += 1
										st.session_state.answer_submitted[qidx] = {"submitted": True, "is_correct": True, "user": user}
									else:
										st.session_state.answer_submitted[qidx] = {"submitted": True, "is_correct": False, "user": user}
										# 将错误加入错题本
										entry = {"english": cur.get('english'), "chinese": cur.get('chinese', '')}
										if entry not in st.session_state.wrongbook:
											st.session_state.wrongbook.append(entry)
											save_wrongbook(st.session_state.wrongbook)
									st.rerun()

							else:
								# 已提交，显示与选择题相似的结果页面（高亮用户答案与正确答案）
								user_ans = st.session_state.answer_submitted[qidx].get("user", "")
								if is_correct:
									st.success("🎉 太棒了！回答正确！")
								else:
									st.error("❌ 很遗憾，回答错误！")
								# 显示正确答案与用户答案对比
								st.markdown("---")
								st.markdown(f"### ✅ 正确答案： {cur.get('english')}")
								st.markdown(f"**你的回答：** {user_ans}")
								# 若回答错误，进一步强调并在视觉上区分
								if not is_correct:
									st.markdown("### 📌 建议复习：")
									st.info(f"请复习：{cur.get('english')} — {cur.get('chinese','')}")
								# 同样将错误加入错题本（若尚未加入） — 已在提交时处理
								# 操作按钮：学会了（移除错题并下一题） / 结束考察
								btns = st.columns([1, 1])
								if btns[0].button("学会了，下一道", key=f"learned_{qidx}"):
									entry = {"english": cur.get('english'), "chinese": cur.get('chinese','')}
									st.session_state.wrongbook = [x for x in st.session_state.wrongbook if x != entry]
									save_wrongbook(st.session_state.wrongbook)
									if qidx < st.session_state.test_progress["total"] - 1:
										st.session_state.test_progress["asked"] = qidx + 1
										st.rerun()
									else:
										st.info("已经是最后一题，考察结束或提交当前结果。")
								if btns[1].button("结束考察", key=f"end_after_{qidx}"):
									# 结束本次考察：展示结果、气球、鼓励语，并清空当前考察队列
									total = st.session_state.test_progress.get("total", 0)
									correct = st.session_state.test_progress.get("correct", 0)
									st.balloons()
									st.success(f"已完成考察：{correct}/{total} 正确")
									encouragements = [
										"干得好！继续保持，每天进步一点。",
										"坚持就是胜利，你离目标又近了一步！",
										"优秀！复习错题会让你更稳固记忆。",
									]
									st.info(random.choice(encouragements))
									st.session_state.test_queue = []
									st.session_state.test_progress = {"asked": 0, "correct": 0, "total": 0}
									st.rerun()
						
						else:
							# 选择题模式
							st.markdown(f"**请选择 '{cur.get('english','')}' 对应的中文意思**")
							
							# 检查是否已经为这道题生成了选项（避免每次重新打乱）
							if qidx not in st.session_state.choice_options:
								# 生成选项：1个正确答案 + 3个干扰项
								# 从整个单词库中随机选择干扰项（排除当前单词）
								all_words = [w for w in st.session_state.words if w != cur]
								wrong_options = random.sample(all_words, min(3, len(all_words)))
								
								# 合并正确选项和干扰项，然后随机打乱
								options = [{"english": cur["english"], "chinese": cur["chinese"], "is_correct": True}] + \
								          [{"english": w["english"], "chinese": w["chinese"], "is_correct": False} for w in wrong_options]
								random.shuffle(options)  # 随机打乱所有选项
								
								# 保存到 session_state
								st.session_state.choice_options[qidx] = options
							else:
								# 使用已保存的选项
								options = st.session_state.choice_options[qidx]
							
							# 检查当前题目是否已经提交答案
							submitted = st.session_state.answer_submitted.get(qidx, {}).get("submitted", False)
							is_correct_answer = st.session_state.answer_submitted.get(qidx, {}).get("is_correct", False)
							
							if not submitted:
								# 还没有提交答案，显示题目和选项
								option_labels = [f"{chr(65+i)}. {opt['chinese']}" for i, opt in enumerate(options)]
								selected = st.radio("请选择正确答案", options=option_labels, key=f"choice_{qidx}")
								
								# 添加上一题和下一题按钮（浏览功能）
								nav_cols = st.columns([1, 1, 2])
								if nav_cols[0].button("⏮️ 上一题", key="nav_prev_choice", disabled=(qidx == 0)):
									st.session_state.test_progress["asked"] = max(0, qidx - 1)
									st.rerun()
								if nav_cols[1].button("下一题", key="nav_next_choice"):
									if qidx < st.session_state.test_progress["total"] - 1:
										st.session_state.test_progress["asked"] = qidx + 1
										st.rerun()
									else:
										st.info("已经是最后一题了，请提交答案或结束考察")
								
								if st.button("✅ 提交答案", key=f"submit_choice_{qidx}"):
									# 获取用户选择的索引
									selected_idx = option_labels.index(selected)
									is_correct = options[selected_idx]["is_correct"]
									
									# 保存答题状态
									st.session_state.answer_submitted[qidx] = {
										"submitted": True,
										"is_correct": is_correct,
										"selected_idx": selected_idx
									}
									
									if is_correct:
										st.session_state.test_progress["correct"] += 1
									
									st.rerun()
							
							else:
								# 已经提交答案，显示结果和解析
								selected_idx = st.session_state.answer_submitted[qidx]["selected_idx"]
								
								if is_correct_answer:
									# 回答正确
									st.success("🎉 太棒了！回答正确！")
								else:
									# 回答错误，显示详细解析
									st.error("❌ 很遗憾，回答错误！")
									
									# 找到正确答案的选项
									correct_option = None
									for opt in options:
										if opt["is_correct"]:
											correct_option = opt
											break
									
									# 显示正确答案（使用醒目的格式）
									st.markdown("---")
									st.markdown(f"### ✅ 正确答案")
									st.success(f"**{correct_option['chinese']}** ({correct_option['english']})")
									
									# 显示所有选项的详细信息（使用颜色区分）
									st.markdown("### 📊 所有选项详情")
									for i, opt in enumerate(options):
										if opt["is_correct"]:
											# 正确答案 - 绿色背景
											st.success(f"**{chr(65+i)}. {opt['chinese']}** ({opt['english']}) ✅ 正确答案")
										elif i == selected_idx:
											# 用户选择的错误答案 - 红色背景
											st.error(f"{chr(65+i)}. {opt['chinese']} ({opt['english']}) ❌ 你的选择")
										else:
											# 其他干扰项 - 普通显示
											st.info(f"{chr(65+i)}. {opt['chinese']} ({opt['english']})")
									
									# 加入错题本
									entry = {"english": cur.get('english'), "chinese": cur.get('chinese','')}
									if entry not in st.session_state.wrongbook:
										st.session_state.wrongbook.append(entry)
										save_wrongbook(st.session_state.wrongbook)
									
									st.markdown("---")
								
								# 显示下一题按钮
								if st.button("➡️ 下一题", key=f"next_after_submit_{qidx}"):
									st.session_state.test_progress["asked"] += 1
									st.rerun()
					else:
						# 完成考察
						total = st.session_state.test_progress["total"]
						correct = st.session_state.test_progress["correct"]
						st.balloons()
						st.success(f"已完成考察：{correct}/{total} 正确")
						# 语言鼓励
						encouragements = [
							"干得好！继续保持，每天进步一点。",
							"坚持就是胜利，你离目标又近了一步！",
							"优秀！复习会让你更稳固记忆。",
						]
						st.info(random.choice(encouragements))
						if st.button("结束并清空本次考察", key="end_test"):
							st.session_state.test_queue = []
							st.session_state.test_progress = {"asked": 0, "correct": 0, "total": 0}

	with right:
		st.header("AI 助记（聊天）")
		st.write("仅限关于单词释义、拼写、例句或记忆方法的问题。")
		for msg in st.session_state.chat_history:
			if msg[0] == "user":
				st.markdown(f"**你：** {msg[1]}")
			else:
				st.markdown(f"**AI：** {msg[1]}")

		user_input = st.text_input("向 AI 提问（例如：'apple 的记忆方法'）", key="ai_input")
		if st.button("发送", key="ai_send") and user_input.strip():
			st.session_state.chat_history.append(("user", user_input))
			# prepare messages from chat_history for context (system already applied in send_ai_query)
			messages = []
			for role, text in st.session_state.chat_history:
				messages.append({"role": "user" if role == "user" else "assistant", "content": text})
			resp = send_ai_query(messages, user_input)
			st.session_state.chat_history.append(("ai", resp))
			st.rerun()


if __name__ == "__main__":
	main()