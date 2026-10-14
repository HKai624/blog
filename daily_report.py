"""WorkMate 每日早报：天气 + 智能建议 + AI 新闻 + LLM 摘要"""

import os
import sys
import json
import time
import smtplib
import logging
from email.mime.text import MIMEText
from email.utils import formataddr
from datetime import datetime

import requests

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── 日志 ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ── 配置 ──────────────────────────────────────────────────────
QQ_ADDRESS = os.getenv("QQ_EMAIL_ADDRESS", "")
QQ_PASSWORD = os.getenv("QQ_EMAIL_PASSWORD", "")
RECIPIENT = os.getenv("QQ_EMAIL_ADDRESS", "")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

# 是否启用 LLM 摘要（可通过环境变量关闭）
ENABLE_LLM_SUMMARY = os.getenv("ENABLE_LLM_SUMMARY", "true").lower() == "true"

# 南宁经纬度
NANNING_LAT = 22.82
NANNING_LON = 108.37

# Open-Meteo WMO 天气码 → 中文
WMO_CODES = {
    0: "晴天", 1: "少云", 2: "多云", 3: "阴天",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "小冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "大阵雨",
    85: "小阵雪", 86: "阵雪",
    95: "雷暴", 96: "雷暴+小冰雹", 99: "雷暴+大冰雹",
}


# ── LLM 调用 ──────────────────────────────────────────────────
def _build_llm_client():
    """延迟导入 OpenAI 客户端，避免本地未安装 openai 时直接崩溃"""
    try:
        from openai import OpenAI
        return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    except ImportError:
        return None


def call_llm(prompt: str, fallback: str = "") -> str:
    """调用 DeepSeek LLM，失败返回 fallback"""
    if not DEEPSEEK_API_KEY:
        logging.warning("未设置 DEEPSEEK_API_KEY，跳过 LLM 调用")
        return fallback

    client = _build_llm_client()
    if client is None:
        logging.warning("openai 库未安装，跳过 LLM 调用")
        return fallback

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
                timeout=15,
            )
            text = resp.choices[0].message.content.strip()
            logging.info(f"LLM 调用成功: {text[:60]}...")
            return text
        except Exception as e:
            logging.warning(f"LLM 调用失败 (第{attempt + 1}次): {e}")
            if attempt < 1:
                time.sleep(2)

    return fallback


# ── 天气 ──────────────────────────────────────────────────────
def get_nanning_weather() -> dict:
    """Open-Meteo 获取南宁天气原始数据"""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={NANNING_LAT}&longitude={NANNING_LON}"
        f"&current=temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
        f"&timezone=Asia/Shanghai"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        cur = data.get("current", {})
        daily = data.get("daily", {})
        return {
            "temp": cur.get("temperature_2m", "?"),
            "weather_code": cur.get("weather_code", -1),
            "weather_text": WMO_CODES.get(cur.get("weather_code", -1), "未知"),
            "humidity": cur.get("relative_humidity_2m", "?"),
            "wind": cur.get("wind_speed_10m", "?"),
            "high": daily.get("temperature_2m_max", ["?"])[0],
            "low": daily.get("temperature_2m_min", ["?"])[0],
            "rain_prob": daily.get("precipitation_probability_max", [0])[0] or 0,
        }
    except Exception as e:
        logging.error(f"天气获取失败: {e}")
        return {}


def generate_weather_advice(weather: dict) -> str:
    """调用 LLM 生成天气行动建议"""
    if not weather or not ENABLE_LLM_SUMMARY:
        return _fallback_weather_advice(weather)

    prompt = f"""你是生活助手。根据以下天气数据，用一句简短中文给出今日出行建议（15字以内，带emoji）。

天气：{weather['weather_text']}
温度：{weather['temp']}℃（最高{weather['high']}℃ / 最低{weather['low']}℃）
湿度：{weather['humidity']}%
风速：{weather['wind']}km/h
降雨概率：{weather['rain_prob']}%

规则：
- 有雨/降雨概率>30% → 提醒带伞 ☔
- 最高温>35℃ → 提醒防暑 🥵
- 最低温<10℃ → 提醒保暖 🧣
- 大风>30km/h → 提醒防风 💨
- 其他情况 → 根据天气自由发挥
只输出一句话，不要多余内容。"""

    result = call_llm(prompt)
    return result if result else _fallback_weather_advice(weather)


def _fallback_weather_advice(weather: dict) -> str:
    """无 LLM 时的天气建议降级"""
    if not weather:
        return "天气数据获取失败"
    code = weather["weather_code"]
    temp = weather["temp"]
    wind = weather["wind"]
    rain = weather["rain_prob"]

    parts = []
    if code in (61, 63, 65, 80, 81, 82, 95, 96, 99) or rain > 30:
        parts.append("记得带伞 ☔")
    if isinstance(temp, (int, float)):
        if temp > 35:
            parts.append("注意防暑 🥵")
        elif temp < 10:
            parts.append("注意保暖 🧣")
    if isinstance(wind, (int, float)) and wind > 30:
        parts.append("注意防风 💨")
    if not parts:
        parts.append("天气不错，适合出行 🌤️")
    return "，".join(parts) + "。"


def format_weather_summary(weather: dict) -> str:
    """格式化天气摘要文本"""
    if not weather:
        return "天气数据获取失败"
    return (
        f"{weather['weather_text']}，当前{weather['temp']}℃，"
        f"最高{weather['high']}℃ / 最低{weather['low']}℃，"
        f"湿度{weather['humidity']}%，风速{weather['wind']}km/h，"
        f"降雨概率{weather['rain_prob']}%"
    )


# ── AI 新闻 ───────────────────────────────────────────────────
def generate_news_summary(item: dict) -> dict:
    """为单条新闻调用 LLM 生成摘要 + 重要性说明"""
    if not ENABLE_LLM_SUMMARY:
        return _fallback_news_summary(item)

    title = item["title"]
    source = item["source"]

    prompt = f"""你是 AI 领域资深编辑。阅读以下新闻/论文标题，用 JSON 格式回复（只输出 JSON，不要其他内容）：

标题：{title}
来源：{source}

格式：
{{"summary": "一句话中文摘要，不超过30字", "importance": "为何对AI从业者重要，不超过20字"}}"""

    try:
        result = call_llm(prompt)
        if result:
            parsed = json.loads(result)
            return {
                "summary": parsed.get("summary", title[:30]),
                "importance": parsed.get("importance", "关注 AI 前沿动态"),
            }
    except (json.JSONDecodeError, Exception) as e:
        logging.warning(f"新闻摘要解析失败: {e}")

    return _fallback_news_summary(item)


def _fallback_news_summary(item: dict) -> dict:
    """无 LLM 时的新闻摘要降级：截取标题"""
    title = item["title"]
    return {
        "summary": title[:40] + ("..." if len(title) > 40 else ""),
        "importance": "AI 领域最新动态",
    }


def fetch_and_summarize_news() -> tuple[list[dict], bool]:
    """抓取并摘要新闻，返回 (新闻列表, 是否全部成功)"""
    try:
        from src.news_fetcher import fetch_all_news
        raw_news = fetch_all_news(max_total=5)
    except Exception as e:
        logging.error(f"新闻抓取失败: {e}")
        return [], False

    if not raw_news:
        logging.warning("所有新闻源均不可用")
        return [], False

    enriched = []
    all_ok = True
    for item in raw_news:
        try:
            summary_data = generate_news_summary(item)
            item["summary"] = summary_data["summary"]
            item["importance"] = summary_data["importance"]
            enriched.append(item)
        except Exception:
            all_ok = False
            item["summary"] = item["title"][:40]
            item["importance"] = "AI 领域最新动态"
            enriched.append(item)

    return enriched, all_ok


# ── 邮件发送 ──────────────────────────────────────────────────
def send_email(content: str, to_addr: str):
    smtp_server = "smtp.qq.com"
    port = 587

    msg = MIMEText(content, "plain", "utf-8")
    msg["From"] = formataddr(("WorkMate AI早报", QQ_ADDRESS))
    msg["To"] = formataddr(("你", to_addr))
    msg["Subject"] = f"AI 早报 | 南宁天气 ({datetime.now().strftime('%m-%d')})"

    try:
        server = smtplib.SMTP(smtp_server, port, timeout=15)
        server.starttls()
        server.login(QQ_ADDRESS, QQ_PASSWORD)
        server.sendmail(QQ_ADDRESS, [to_addr], msg.as_string())
        logging.info(f"邮件发送成功 -> {to_addr}")
    except Exception as e:
        logging.error(f"邮件发送失败: {e}")
    finally:
        try:
            server.quit()
        except Exception:
            pass


# ── 邮件正文模板 ──────────────────────────────────────────────
def build_email_body(
    date_str: str,
    weather: dict,
    weather_advice: str,
    news_items: list[dict],
    news_ok: bool,
) -> str:
    lines = [
        f"☀️ WorkMate AI 早报 · {date_str}",
        "",
        "━" * 36,
        "",
        "🌤️ 南宁今日天气",
        f"   {format_weather_summary(weather)}",
        f"   💡 {weather_advice}",
        "",
        "━" * 36,
        "",
        "🤖 今日 AI 精选",
        "",
    ]

    if news_items:
        for i, item in enumerate(news_items, 1):
            lines.append(f"  {i}. {item['title']}")
            lines.append(f"     📝 {item['summary']}")
            lines.append(f"     💡 {item['importance']}")
            lines.append(f"     🔗 {item['link']}")
            lines.append(f"     📰 {item['source']}")
            lines.append("")
    else:
        lines.append("  今日 AI 新闻源暂时无法获取，稍后重试。")
        lines.append("")

    if not news_ok and news_items:
        lines.append("   ⚠️ 部分新闻摘要生成失败，已显示原始标题。")
        lines.append("")

    lines.extend([
        "━" * 36,
        "本邮件由 WorkMate Agent v1.0 自动生成 · DeepSeek 驱动",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ])

    return "\n".join(lines)


# ── 主流程 ────────────────────────────────────────────────────
def daily_job():
    logging.info("执行每日早报...")

    # 1. 天气
    weather = get_nanning_weather()
    weather_advice = generate_weather_advice(weather)
    logging.info(f"天气: {weather.get('weather_text', '?')} | 建议: {weather_advice}")

    # 2. AI 新闻
    news_items, news_ok = fetch_and_summarize_news()
    logging.info(f"AI 新闻: {len(news_items)} 条, LLM摘要全部成功={news_ok}")

    # 3. 组装邮件
    today = datetime.now().strftime("%Y年%m月%d日")
    body = build_email_body(today, weather, weather_advice, news_items, news_ok)

    # 4. 发送
    send_email(body, RECIPIENT)


def main():
    if not QQ_ADDRESS or QQ_ADDRESS == "your_qq@qq.com":
        logging.error("未设置 QQ_EMAIL_ADDRESS")
        sys.exit(1)
    if not QQ_PASSWORD:
        logging.error("未设置 QQ_EMAIL_PASSWORD")
        sys.exit(1)

    if not DEEPSEEK_API_KEY:
        logging.warning("未设置 DEEPSEEK_API_KEY，将使用降级模式（无 LLM 摘要）")

    is_daemon = len(sys.argv) > 1 and sys.argv[1] == "--daemon"

    if is_daemon:
        import schedule
        from datetime import timedelta

        job = schedule.every().day.at("07:00").do(daily_job)
        logging.info("定时器已启动，每天 07:00 执行...")

        now = datetime.now()
        target = now.replace(hour=7, minute=0, second=0, microsecond=0)
        if now > target:
            job.next_run = target + timedelta(days=1)
            logging.info("已过 07:00，首次推送将在明天 07:00 执行")

        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        daily_job()


if __name__ == "__main__":
    main()
