import streamlit as st
import pandas as pd
import json
import os
import random
from typing import List, Dict

try:
	import openai
except Exception:
	openai = None

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
			df = pd.read_csv(pd.compat.StringIO(s), sep=sep)
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


def send_ai_query(messages: List[Dict], user_input: str) -> str:
	"""简单过滤：只允许与单词/记忆相关的问题，否则拒绝回答。若配置了 OpenAI key 则调用 API。"""
	allowed_keywords = ["单词", "记忆", "拼写", "词义", "意思", "例句", "近义", "联想", "记法", "词根", "词缀", "如何记", "怎么记"]
	low = user_input.lower()
	if not any(k in low for k in allowed_keywords) and len(low.split()) > 5:
		return "抱歉，我只能回答与单词和背记相关的问题。请把问题限定为单词的释义、拼写、例句或记忆方法等。"

	if openai is None:
		# 本地简单应答：如果看起来像是单词查询，尝试返回一个模板回答
		tokens = user_input.split()
		word = tokens[0]
		return f"（本地模拟回答）关于 '{word}'：建议先记词义，再联想一个短句；例如：'{word}' — 例句：This is a sample sentence containing {word}."

	key = os.environ.get("OPENAI_API_KEY")
	if not key:
		return "未配置 OpenAI API Key，无法使用 AI 功能。请设置环境变量 OPENAI_API_KEY 或在界面使用本地回答模式。"

	openai.api_key = key
	system_prompt = "你是一个只回答关于单词、词义、拼写、例句以及记忆方法的助教，不回答其他内容。若用户问与单词无关的问题，则简短拒绝并引导回到单词主题。用中文回答。"
	try:
		chat = openai.ChatCompletion.create(
			model="gpt-3.5-turbo",
			messages=[{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": user_input}],
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

	with left:
		st.header("操作面板")
		uploaded = st.file_uploader("导入单词表（CSV: english,chinese 或 每行 english<TAB>chinese）", type=["csv", "txt"])
		if uploaded is not None:
			try:
				rows = parse_wordfile(uploaded.read())
				if rows:
					st.session_state.words = rows
					st.session_state.index = 0
					st.success(f"已导入 {len(rows)} 个单词")
			except Exception as e:
				st.error(f"导入失败：{e}")

		if st.button("载入示例单词"):
			sample = load_sample()
			if sample:
				st.session_state.words = sample
				st.session_state.index = 0
				st.success(f"已载入示例，共 {len(sample)} 个单词")
			else:
				st.warning("未找到示例文件 sample_words.csv")

		st.write("---")
		st.radio("选择模式", options=["memory", "test"], index=0 if st.session_state.mode == "memory" else 1, key="mode_radio")
		st.session_state.mode = st.session_state.mode_radio

		st.write("错题本：")
		if st.button("查看/清空错题本"):
			st.session_state.wrongbook = load_wrongbook()
			if st.session_state.wrongbook:
				if st.button("清空错题本确认"):
					st.session_state.wrongbook = []
					save_wrongbook([])
					st.success("已清空错题本")
			else:
				st.info("当前错题本为空")

		st.write(f"错题数量：{len(st.session_state.wrongbook)}")

	with mid:
		if st.session_state.mode == "memory":
			st.header("记忆模式")
			if not st.session_state.words:
				st.info("请在左侧导入单词或载入示例")
			else:
				w = st.session_state.words[st.session_state.index % len(st.session_state.words)]
				st.subheader(w.get("english", ""))
				st.write(w.get("chinese", ""))
				cols = st.columns(3)
				if cols[0].button("上一个"):
					st.session_state.index = (st.session_state.index - 1) % len(st.session_state.words)
				if cols[1].button("下一个"):
					st.session_state.index = (st.session_state.index + 1) % len(st.session_state.words)
				if cols[2].button("标记为不会（加入错题本）"):
					# add to wrongbook
					entry = {"english": w["english"], "chinese": w.get("chinese", "")}
					if entry not in st.session_state.wrongbook:
						st.session_state.wrongbook.append(entry)
						save_wrongbook(st.session_state.wrongbook)
						st.success("已加入错题本")

		else:
			st.header("考察模式")
			if not st.session_state.words and not st.session_state.wrongbook:
				st.info("请先导入单词或加入错题本")
			else:
				test_source = st.selectbox("选择考察词库", options=["全部单词", "错题本"], index=0)
				total_available = len(st.session_state.words) if test_source == "全部单词" else len(st.session_state.wrongbook)
				n = st.number_input("考察数量", min_value=1, max_value=max(1, total_available), value=min(10, max(1, total_available)))
				if st.button("开始考察"):
					pool = st.session_state.words if test_source == "全部单词" else st.session_state.wrongbook
					if not pool:
						st.warning("所选词库为空")
					else:
						st.session_state.test_queue = random.sample(pool, min(n, len(pool)))
						st.session_state.test_progress = {"asked": 0, "correct": 0, "total": len(st.session_state.test_queue)}

				if st.session_state.test_queue:
					qidx = st.session_state.test_progress["asked"]
					if qidx < st.session_state.test_progress["total"]:
						cur = st.session_state.test_queue[qidx]
						st.write(f"第 {qidx+1} / {st.session_state.test_progress['total']} 题")
						st.markdown(f"**中文提示：** {cur.get('chinese','')} ")
						ans = st.text_input("请输入英文拼写（不区分大小写）", key=f"ans_{qidx}")
						if st.button("提交答案", key=f"submit_{qidx}"):
							user = ans.strip().lower()
							target = cur.get("english"," ").strip().lower()
							if user == target:
								st.session_state.test_progress["correct"] += 1
								st.success("回答正确！")
							else:
								st.error(f"回答错误，正确答案：{cur.get('english')}")
								entry = {"english": cur.get('english'), "chinese": cur.get('chinese','')}
								if entry not in st.session_state.wrongbook:
									st.session_state.wrongbook.append(entry)
									save_wrongbook(st.session_state.wrongbook)
							st.session_state.test_progress["asked"] += 1
							st.experimental_rerun()
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
							"优秀！复习错题会让你更稳固记忆。",
						]
						st.info(random.choice(encouragements))
						if st.button("结束并清空本次考察"):
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
			st.experimental_rerun()


if __name__ == "__main__":
	main()

