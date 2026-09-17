"""Cikti uzayini bilerek kisitlayan enum'lar (spec §4).

Modelin `greater_than`, `equals`, `>`, `after` gibi alternatif temsiller
uretmesi engellenir; dogal dildeki tum varyasyon bu kanonik degerlere iner.
"""

from enum import Enum


class QueryOperation(str, Enum):
    SELECT = "select"
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"


class FilterOperator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    BETWEEN = "between"
    IN = "in"
    CONTAINS = "contains"


class OrderDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


class QueryStatus(str, Enum):
    """V2 red/aciklama-isteme durumu (CLAUDE_V2_MIGRATION.md).

    `VALID` disindaki her durum, modelin sorguyu dogrudan yapilandirilmis
    bir sekilde cevaplamak yerine bir aciklama/red gerektigini isaret
    ettigi anlamina gelir -- boylece belirsiz veya celiskili istekler
    sessizce (ve yanlislikla) gecerli bir sorguya donusturulmez.
    """

    VALID = "valid"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
