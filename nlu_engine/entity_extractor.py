import re

class EntityExtractor:
    def __init__(self):

        # Transaction IDs
        self.txn_patterns = [
            re.compile(r'\b(?:txn|transaction)\s*[:\-]?\s*[A-Z0-9\-]{4,20}\b', re.IGNORECASE),
            re.compile(r'\bUTR[: ]?[A-Z0-9]{6,20}\b', re.IGNORECASE)
        ]

        # Account Numbers
        self.account_patterns = [
            re.compile(r'\b(?:account|acct|a\/c|ac)\s*(no\.?|number)?\s*[:\-]?\s*\d{6,20}\b', re.IGNORECASE),
            re.compile(r'\bto\s+account\s+\d{6,20}\b', re.IGNORECASE)
        ]

        # Amounts WITH currency
        self.amount_patterns_currency = [
            re.compile(r'\b(?:rs|inr|usd|eur)\s*[\.:]?\s*\d+(?:,\d{3})*(?:\.\d+)?\b', re.IGNORECASE),
            re.compile(r'[₹$€]\s*\d+(?:,\d{3})*(?:\.\d+)?\b')
        ]

        # Amounts WITHOUT currency
        self.amount_plain = re.compile(
            r'\b(?:transfer|send|withdraw|deposit|pay|give|move)\s+(\d+(?:,\d{3})*(?:\.\d+)?)\b',
            re.IGNORECASE
        )

        # From / To accounts by NAME
        self.from_account = re.compile(r'\bfrom\s+(savings|checking|salary|current|wallet)\b', re.IGNORECASE)
        self.to_account = re.compile(r'\bto\s+(savings|checking|salary|current|wallet)\b', re.IGNORECASE)

    def _clean(self, text):
        return text.replace(",", "").strip().upper()

    def extract(self, text: str):
        results = []

        # ------------ TRANSACTION IDs ------------
        for pattern in self.txn_patterns:
            for m in pattern.finditer(text):
                results.append(("txn", m.group(0)))

        # ------------ ACCOUNT NUMBERS ------------
        for pattern in self.account_patterns:
            for m in pattern.finditer(text):
                # extract only the digits
                acc = re.findall(r'\d{6,20}', m.group(0))
                if acc:
                    results.append(("account", acc[0]))

        # ------------ AMOUNTS WITH CURRENCY ------------
        for pattern in self.amount_patterns_currency:
            for m in pattern.finditer(text):
                results.append(("amount", self._clean(m.group(0))))

        # ------------ AMOUNTS WITHOUT CURRENCY ------------
        for m in self.amount_plain.finditer(text):
            results.append(("amount", self._clean(m.group(1))))

        # ------------ FROM ACCOUNT NAME ------------
        m = self.from_account.search(text)
        if m:
            results.append(("from_account", m.group(1).lower()))

        # ------------ TO ACCOUNT NAME ------------
        m = self.to_account.search(text)
        if m:
            results.append(("to_account", m.group(1).lower()))

        # ---------- REMOVE DUPLICATES ----------
        unique = []
        seen = set()

        for etype, value in results:
            key = (etype, value)
            if key not in seen:
                unique.append((etype, value))
                seen.add(key)

        return unique


def extract(text):
    return EntityExtractor().extract(text)
