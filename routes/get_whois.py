import re

import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, status
from loguru import logger

from lib.constants import DEFAULT_HTTP_TIMEOUT_SECONDS
from lib.utils import error_response, success_response

router = APIRouter()


@router.get("/get-whois")
def get_whois(domain: str):
    if not domain or not domain.strip():
        return error_response("Domain is required", status.HTTP_400_BAD_REQUEST)

    try:
        url = f"https://www.whois.com/whois/{domain}"

        response = requests.get(url, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        blocks = soup.find_all("div", class_="df-block")

        data = {}

        emails = []

        countries = []

        for block in blocks:
            heading_tag = block.find("div", class_="df-heading")

            if heading_tag:
                block_data = {}

                heading = heading_tag.text.strip()

                rows = block.find_all("div", class_="df-row")

                for row in rows:
                    label_tag = row.find("div", class_="df-label")

                    value_tag = row.find("div", class_="df-value")

                    if label_tag and value_tag:
                        email_img = value_tag.find("img", class_="email")

                        if email_img and email_img.get("src"):
                            logger.warning(
                                f"Email found in image format, skipping: {email_img.get('src')}"
                            )

                            continue
                        else:
                            value = value_tag.get_text(separator=", ").strip()

                            key = label_tag.text.strip().rstrip(":")

                            block_data[key] = value

                            found_emails = re.findall(
                                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                                value,
                            )

                            emails.extend(found_emails)

                            if key.lower() == "country":
                                countries.append(value)

                data[heading] = block_data

        emails = list(set(emails))

        countries = list(set(countries))

        logger.info(f"WHOIS data fetched successfully for {domain}")

        return success_response(
            "WHOIS data fetched successfully",
            data={
                "emails": emails,
                "countries": countries,
                "whois": data,
            },
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Error fetching WHOIS data for {domain}: {e}")
        return error_response("Failed to process whois", status.HTTP_502_BAD_GATEWAY)
