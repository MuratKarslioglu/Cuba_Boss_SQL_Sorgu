"""V1/V2 sorgu sozlesmesi (spec §3, §6; V2: CLAUDE_V2_MIGRATION.md).

Bu sema kasten sade tutulmustur. Isverenin gercek JSON sozlesmesi geldiginde
burasi degisecek; bu nedenle `target` ve filtre `field` degerleri enum degil
serbest string'tir. Izin verilen isim listesi ayri bir katmanda tutulur
(`app.schemas.vocabulary`) ve yalnizca dataset dogrulamasinda uygulanir.

V2 eklentisi -- `status`/`clarification`:

Gercek dunya girdileri her zaman dogrudan yapilandirilmis bir sorguya
cevrilebilir degildir (belirsiz, celiskili, veya yetersiz bilgi iceren
istekler). `status` alani modelin bu durumu ACIKCA isaretlemesini saglar;
boylece belirsiz bir istek sessizce (ve yanlislikla) gecerli bir sorguya
donusturulmez (spec'in "sessizce onarma" yasagi V2'ye de tasinir).

Geriye donuk uyumluluk: `status`/`clarification` varsayilan degerlere
sahiptir (`VALID`/`None`), boylece bu alanlari hic icermeyen eski V1
verisi degismeden gecerli kalmaya devam eder.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.enums import FilterOperator, OrderDirection, QueryOperation, QueryStatus
from app.schemas.semantics import find_filter_contradictions

# Spec §9: modelin egitim boyunca gordugu anahtar sirasi sabittir.
# V2: `status`/`clarification` sona eklenir (gercek V2 dataset'indeki
# gozlemlenen anahtar sirasiyla birebir).
CANONICAL_KEY_ORDER: tuple[str, ...] = (
    "operation",
    "target",
    "filters",
    "group_by",
    "order_by",
    "limit",
    "status",
    "clarification",
)

# Liste degeri bekleyen operatorler (spec §21).
_LIST_VALUED_OPERATORS = {FilterOperator.BETWEEN, FilterOperator.IN}


class _StrictModel(BaseModel):
    """Bilinmeyen anahtarlari reddeden taban.

    Spec §16/§37: model ciktisi sessizce onarilmaz. Model uydurma bir alan
    uretirse bu sessizce yutulmak yerine acik bir dogrulama hatasi olur.
    """

    model_config = ConfigDict(extra="forbid")


def _require_non_empty(value: str, label: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} bos olamaz")
    return stripped


class FilterCondition(_StrictModel):
    field: str
    operator: FilterOperator
    value: Any

    @field_validator("field")
    @classmethod
    def _validate_field(cls, v: str) -> str:
        return _require_non_empty(v, "field")

    @model_validator(mode="after")
    def _validate_value_shape(self) -> FilterCondition:
        is_list = isinstance(self.value, list)

        if self.operator is FilterOperator.BETWEEN:
            if not is_list or len(self.value) != 2:
                raise ValueError("'between' operatoru tam olarak 2 elemanli bir liste bekler")
        elif self.operator is FilterOperator.IN:
            if not is_list or len(self.value) == 0:
                raise ValueError("'in' operatoru bos olmayan bir liste bekler")
        elif is_list:
            raise ValueError(f"'{self.operator.value}' operatoru liste degeri kabul etmez")

        if self.value is None:
            raise ValueError("filtre degeri None olamaz")

        return self


class OrderBy(_StrictModel):
    field: str
    direction: OrderDirection

    @field_validator("field")
    @classmethod
    def _validate_field(cls, v: str) -> str:
        return _require_non_empty(v, "order_by.field")


class QueryRequest(_StrictModel):
    # V2: bir istek dogrudan cevaplanamiyorsa (belirsiz/celiskili/yetersiz
    # bilgi) operation/target null kalir -- bkz. `_validate_status_contract`.
    operation: QueryOperation | None = None
    target: str | None = None
    filters: list[FilterCondition] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    order_by: OrderBy | None = None
    limit: int | None = Field(default=None, gt=0)
    status: QueryStatus = QueryStatus.VALID
    clarification: str | None = None

    @field_validator("target")
    @classmethod
    def _validate_target(cls, v: str | None) -> str | None:
        return v if v is None else _require_non_empty(v, "target")

    @field_validator("clarification")
    @classmethod
    def _validate_clarification(cls, v: str | None) -> str | None:
        return v if v is None else _require_non_empty(v, "clarification")

    @field_validator("group_by")
    @classmethod
    def _validate_group_by(cls, v: list[str]) -> list[str]:
        return [_require_non_empty(item, "group_by ogesi") for item in v]

    @model_validator(mode="after")
    def _validate_status_contract(self) -> QueryRequest:
        """V2 sozlesme kurali (CLAUDE_V2_MIGRATION.md, schema_v2.json):

            status == VALID     -> operation VE target dolu, clarification bos
            status != VALID     -> operation VE target bos, clarification dolu

        Bu, modelin "valid" deyip yine de operation/target vermemesi veya
        "ambiguous" deyip tam bir sorgu uretmesi gibi ic-tutarsiz, sessizce
        yanlis yorumlanabilecek ciktilari acikca reddeder.
        """
        if self.status is QueryStatus.VALID:
            if self.operation is None or self.target is None:
                raise ValueError("status='valid' iken operation ve target zorunludur")
            if self.clarification is not None:
                raise ValueError("status='valid' iken clarification None olmalidir")
        else:
            if self.operation is not None or self.target is not None:
                raise ValueError(
                    f"status={self.status.value!r} iken operation ve target None olmalidir"
                )
            if self.clarification is None:
                raise ValueError(f"status={self.status.value!r} iken clarification zorunludur")
        return self

    @model_validator(mode="after")
    def _validate_filters_are_consistent(self) -> QueryRequest:
        """Yapisal olarak gecerli ama mantiksal olarak celisen filtreleri
        reddeder (bkz. `app.schemas.semantics`).

        Ornek: `company eq 'ABC'` ve `company eq 'XYZ'` semaya uyar ama
        gercek bir SQL'e cevrildiginde hicbir satirla eslesmez. Bu, gercek
        bir model calistirmasinda gozlemlenen bir hata sinifidir.
        """
        contradictions = find_filter_contradictions(self.filters)
        if contradictions:
            details = "; ".join(c.message for c in contradictions)
            raise ValueError(f"filtreler mantiksal olarak celisiyor: {details}")
        return self


def to_canonical_json(query: QueryRequest) -> str:
    """Egitim ve degerlendirme icin tek kanonik seri hale getirme (spec §9).

    - `ensure_ascii=False`: Turkce karakterler escape edilmez.
    - Bosluksuz separator: token israfi olmaz.
    - `sort_keys=False`: Pydantic'in alan tanim sirasi korunur.
    """
    payload = query.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def canonicalize_dict(data: dict[str, Any]) -> str:
    """Ham bir dict'i once dogrulayip sonra kanonik JSON'a cevirir."""
    return to_canonical_json(QueryRequest.model_validate(data))
