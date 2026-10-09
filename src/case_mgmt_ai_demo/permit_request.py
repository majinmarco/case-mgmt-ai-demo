"""Permit Request models: a shared core of fields plus the fields of each Request Type.

Field names are English; values (and the labels shown to models and Reviewers) are Spanish.

A field is here when it lets a request evidence a requirement that applies to it: a requirement
the record cannot evidence is judged unmet (CMAWD-18).
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ApplicantType = Literal["privado", "gobierno", "cooperativa", "sin_fines_de_lucro"]
ApplicantRole = Literal["dueno", "optante", "arrendatario", "representante", "otro"]
# The standing documents listed in RC-2023-E Sección 2.1.9.5(b).
StandingDocument = Literal[
    "escritura_publica",
    "contrato_arrendamiento",
    "contrato_opcion_compra",
    "certificacion_registral",
    "declaratoria_herederos",
    "transaccion_terrenos_jp",
    "documentos_persona_juridica",
]
PcocDocument = Literal[
    "planos_sellados",
    "certificacion_valor_obra",
    "determinacion_cumplimiento_ambiental",
    "ubicacion_geografica",
    "foto_aerea",
    "mapa_catastral",
]
# RC-2023-E Sección 2.1.9.3(b): the OGPe guides, or the designer's own method.
WorkValueMethod = Literal["guias_ogpe", "otro_mecanismo"]
# The professionals who may certify plans under RC-2023-E Sección 2.1.9.2(k).
PlansProfessional = Literal["ingeniero", "arquitecto", "agrimensor", "arquitecto_paisajista"]

LABELS: dict[str, str] = {
    "privado": "persona privada",
    "gobierno": "entidad gubernamental",
    "cooperativa": "cooperativa",
    "sin_fines_de_lucro": "entidad sin fines de lucro",
    "dueno": "dueño",
    "optante": "optante",
    "arrendatario": "arrendatario",
    "representante": "representante autorizado",
    "otro": "otro (no es dueño, optante ni arrendatario)",
    "escritura_publica": "escritura pública",
    "contrato_arrendamiento": "contrato de arrendamiento",
    "contrato_opcion_compra": "contrato de opción de compra",
    "certificacion_registral": "certificación registral",
    "declaratoria_herederos": "declaratoria de herederos",
    "transaccion_terrenos_jp": "transacción de terrenos aprobada por la JP",
    "documentos_persona_juridica": "documentos de existencia y autorización de la persona jurídica",
    "planos_sellados": "planos sellados y firmados por profesional licenciado",
    "certificacion_valor_obra": "certificación del valor estimado de la obra",
    "determinacion_cumplimiento_ambiental": "determinación de cumplimiento ambiental",
    "ubicacion_geografica": "ubicación geográfica del predio",
    "foto_aerea": "foto aérea del predio",
    "mapa_catastral": "mapa catastral",
    "guias_ogpe": "calculado según las guías de la OGPe",
    "otro_mecanismo": "calculado por otro mecanismo del proyectista, no menor que las guías de la OGPe",
    "ingeniero": "ingeniero",
    "arquitecto": "arquitecto",
    "agrimensor": "agrimensor",
    "arquitecto_paisajista": "arquitecto paisajista",
}


def _number(value: Decimal) -> str:
    return f"{value:,}"


class Yards(BaseModel):
    model_config = ConfigDict(extra="forbid")

    front: Decimal = Field(ge=0)
    side: Decimal = Field(ge=0)
    rear: Decimal = Field(ge=0)

    def summary_es(self) -> str:
        return f"{_number(self.front)} m / {_number(self.side)} m / {_number(self.rear)} m"


class DistrictParameters(BaseModel):
    """The parameters of RC-2023-E Sección 2.2.1.3(d), each next to the district's limit."""

    model_config = ConfigDict(extra="forbid")

    lot_area_m2: Decimal = Field(gt=0)
    min_lot_area_m2: Decimal = Field(gt=0)
    dwelling_units: int = Field(ge=0)
    max_dwelling_units: int = Field(ge=0)
    occupation_pct: Decimal = Field(ge=0, le=100)
    max_occupation_pct: Decimal = Field(ge=0, le=100)
    yards_m: Yards
    min_yards_m: Yards
    parking_spaces: int = Field(ge=0)
    required_parking_spaces: int = Field(ge=0)
    loading_spaces: int = Field(ge=0)
    required_loading_spaces: int = Field(ge=0)

    def summary_es(self) -> str:
        yards = self.yards_m
        return "; ".join(
            [
                f"tamaño del solar {_number(self.lot_area_m2)} m² "
                f"(mínimo {_number(self.min_lot_area_m2)} m²)",
                f"densidad {self.dwelling_units} unidad(es) de vivienda "
                f"(máximo {self.max_dwelling_units})",
                f"área de ocupación {_number(self.occupation_pct)}% "
                f"(máximo {_number(self.max_occupation_pct)}%)",
                f"patios delantero {_number(yards.front)} m / laterales {_number(yards.side)} m / "
                f"posterior {_number(yards.rear)} m (mínimos {self.min_yards_m.summary_es()})",
                f"espacios de estacionamiento {self.parking_spaces} "
                f"(requeridos {self.required_parking_spaces})",
                f"espacios de carga y descarga {self.loading_spaces} "
                f"(requeridos {self.required_loading_spaces})",
            ]
        )


class PlansCertification(BaseModel):
    """How the plans are certified, per RC-2023-E Sección 2.1.9.2(k)."""

    model_config = ConfigDict(extra="forbid")

    professional: PlansProfessional
    professional_name: str = Field(min_length=1)
    license_number: str = Field(min_length=1)
    drafted_by: str | None = Field(
        default=None,
        description="Name and licence of who drafted the plans; None when the certifier did.",
    )
    sheets_certified: bool = Field(description="Every sheet carries its required certifications.")
    compliance_certification: bool = Field(
        description="The designer's signed certification that the plans comply."
    )
    file_format: str = Field(min_length=1, description="Digital file format, e.g. 'DXF'.")
    north_oriented: bool

    def summary_es(self) -> str:
        drafter = (
            f"; delineados por {self.drafted_by}"
            if self.drafted_by
            else ", quien también los delineó"
        )
        sheets = (
            "cada hoja incluye" if self.sheets_certified else "no todas las hojas incluyen"
        ) + " las certificaciones específicas requeridas por el Reglamento"
        certification = ("se acompaña" if self.compliance_certification else "no se acompaña") + (
            " el documento de certificación del proyectista de cumplimiento con el Reglamento "
            "Conjunto y los códigos de construcción vigentes"
        )
        orientation = "orientado" if self.north_oriented else "no orientado"
        signed = (
            f"sellados y firmados por {LABELS[self.professional]} licenciado "
            f"{self.professional_name} (licencia {self.license_number}){drafter}"
        )
        return "; ".join(
            [
                signed,
                sheets,
                certification,
                f"archivo digital {self.file_format} {orientation} al norte",
            ]
        )


class PermitRequestCore(BaseModel):
    """Fields every Request Type shares."""

    model_config = ConfigDict(extra="forbid")

    applicant_name: str = Field(min_length=1)
    applicant_type: ApplicantType
    applicant_role: ApplicantRole
    standing_document: StandingDocument | None = None
    municipio: str = Field(min_length=1)
    catastro: str = Field(pattern=r"^\d{3}-\d{3}-\d{3}-\d{2}$")
    catastro_matches_ogpe: bool | None = Field(
        default=None, description="Whether the OGPe catastral records match the Applicant's."
    )
    calificacion: str = Field(min_length=1, description="Zoning district.")

    def core_fields_es(self) -> dict[str, str]:
        standing = LABELS[self.standing_document] if self.standing_document else "ninguno"
        catastro = f"{self.municipio}, {self.catastro}"
        if self.catastro_matches_ogpe is not None:
            matches = "concuerda" if self.catastro_matches_ogpe else "no concuerda"
            catastro += f" (la información catastral {matches} con el sistema de la OGPe)"
        return {
            "Solicitante": f"{self.applicant_name} ({LABELS[self.applicant_type]})",
            "Rol del solicitante": LABELS[self.applicant_role],
            "Documento de legitimación activa": standing,
            "Municipio y número de catastro": catastro,
            "Calificación del distrito": self.calificacion,
        }


class PcocPermitRequest(PermitRequestCore):
    """Permiso de Construcción."""

    request_type: Literal["PCOC"]
    attached_documents: list[PcocDocument]
    work_description: str = Field(min_length=1)
    estimated_work_value: Decimal | None = Field(default=None, gt=0)
    work_value_method: WorkValueMethod | None = None
    floors: int = Field(ge=1)
    height_m: Decimal = Field(gt=0)
    gross_floor_area_m2: Decimal = Field(gt=0)
    declared_fee: Decimal = Field(ge=0)
    district_parameters: DistrictParameters | None = None
    plans_certification: PlansCertification | None = None

    def fields_es(self) -> dict[str, str]:
        """The request as labelled Spanish fields, as shown to models and Reviewers."""
        value = f"${self.estimated_work_value:,}" if self.estimated_work_value else "no declarado"
        if self.estimated_work_value:
            method = LABELS[self.work_value_method] if self.work_value_method else None
            value += f", {method or 'método de cálculo no declarado'}"
        documents = "; ".join(LABELS[d] for d in self.attached_documents) or "ninguno"
        return {
            "Tipo de solicitud": "Permiso de Construcción (PCOC)",
            **self.core_fields_es(),
            "Descripción de la obra": self.work_description,
            "Plantas / altura / área bruta de piso": (
                f"{self.floors} plantas, {self.height_m} m, {self.gross_floor_area_m2} m²"
            ),
            "Valor estimado de la obra": value,
            "Parámetros del distrito": (
                self.district_parameters.summary_es()
                if self.district_parameters
                else "no declarados"
            ),
            "Documentos adjuntos": documents,
            "Certificación de los planos": (
                self.plans_certification.summary_es()
                if self.plans_certification
                else "no declarada"
            ),
            "Derechos declarados": f"${self.declared_fee:,}",
        }


# Becomes a discriminated union on `request_type` when Permiso Único is added.
PermitRequest = PcocPermitRequest
