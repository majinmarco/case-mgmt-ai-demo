"""Permit Request models: a shared core of fields plus the fields of each Request Type.

Field names are English; values (and the labels shown to models and Reviewers) are Spanish.
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
}


class PermitRequestCore(BaseModel):
    """Fields every Request Type shares."""

    model_config = ConfigDict(extra="forbid")

    applicant_name: str = Field(min_length=1)
    applicant_type: ApplicantType
    applicant_role: ApplicantRole
    standing_document: StandingDocument | None = None
    municipio: str = Field(min_length=1)
    catastro: str = Field(pattern=r"^\d{3}-\d{3}-\d{3}-\d{2}$")
    calificacion: str = Field(min_length=1, description="Zoning district.")

    def core_fields_es(self) -> dict[str, str]:
        standing = LABELS[self.standing_document] if self.standing_document else "ninguno"
        return {
            "Solicitante": f"{self.applicant_name} ({LABELS[self.applicant_type]})",
            "Rol del solicitante": LABELS[self.applicant_role],
            "Documento de legitimación activa": standing,
            "Municipio y número de catastro": f"{self.municipio}, {self.catastro}",
            "Calificación del distrito": self.calificacion,
        }


class PcocPermitRequest(PermitRequestCore):
    """Permiso de Construcción."""

    request_type: Literal["PCOC"]
    attached_documents: list[PcocDocument]
    work_description: str = Field(min_length=1)
    estimated_work_value: Decimal | None = Field(default=None, gt=0)
    floors: int = Field(ge=1)
    height_m: Decimal = Field(gt=0)
    gross_floor_area_m2: Decimal = Field(gt=0)
    declared_fee: Decimal = Field(ge=0)

    def fields_es(self) -> dict[str, str]:
        """The request as labelled Spanish fields, as shown to models and Reviewers."""
        value = f"${self.estimated_work_value:,}" if self.estimated_work_value else "no declarado"
        documents = "; ".join(LABELS[d] for d in self.attached_documents) or "ninguno"
        return {
            "Tipo de solicitud": "Permiso de Construcción (PCOC)",
            **self.core_fields_es(),
            "Descripción de la obra": self.work_description,
            "Plantas / altura / área bruta de piso": (
                f"{self.floors} plantas, {self.height_m} m, {self.gross_floor_area_m2} m²"
            ),
            "Valor estimado de la obra": value,
            "Documentos adjuntos": documents,
            "Derechos declarados": f"${self.declared_fee:,}",
        }


# Becomes a discriminated union on `request_type` when Permiso Único is added.
PermitRequest = PcocPermitRequest
