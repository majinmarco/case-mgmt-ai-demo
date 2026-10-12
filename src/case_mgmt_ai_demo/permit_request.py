"""Permit Request models: a shared core of fields plus the fields of each Request Type.

Field names are English; values (and the labels shown to models and Reviewers) are Spanish.

A field is here when it lets a request evidence a requirement that applies to it: a requirement
the record cannot evidence is judged unmet (CMAWD-18).
"""

from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

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
# The site documents of RC-2023-E Sección 2.1.9.2(h)-(i), Sección 3.7.1.2(b), the environmental
# document of 3.7.1.1(b)(2), and the certifications and documents of Sección 4.1.4.1(b).
PermisoUnicoDocument = Literal[
    "foto_aerea",
    "ubicacion_geografica",
    "memorial_explicativo",
    "croquis",
    "fotos_instalacion",
    "exclusion_categorica",
    "antecedentes_penales",
    "no_deuda_asume",
    "no_deuda_crim",
    "planillas_ingresos",
    "planillas_ivu",
    "certificacion_deuda_dh",
    "certificado_incorporacion",
    "numero_patronal",
    "registro_comerciantes",
    "patente_municipal",
    "proyeccion_ventas",
]
License = Literal["sanitaria", "prevencion_incendios", "bebidas_alcoholicas"]
# RC-2023-E Sección 3.7.1.1(f): a Permiso Único needs a ministerial use or an approved variation.
UseAuthorization = Literal["ministerial", "variacion_aprobada", "no_permitido"]
# A new request, or the renewal of an issued Permiso Único (Sección 3.7.1.4).
ApplicationKind = Literal["nueva", "renovacion"]
# The establishments whose Licencia Sanitaria rate is curated from COBROS Anejo A, 30.
EstablishmentType = Literal[
    "no_alimentos", "cafetin", "cafeteria", "restaurante", "otros_alimentos", "supermercado"
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
    "guias_ogpe": "calculado según las guías de la OGPe",
    "otro_mecanismo": "calculado por otro mecanismo del proyectista, no menor que las guías de la OGPe",
    "ingeniero": "ingeniero",
    "arquitecto": "arquitecto",
    "agrimensor": "agrimensor",
    "arquitecto_paisajista": "arquitecto paisajista",
    "memorial_explicativo": "memorial explicativo",
    "croquis": "croquis de la instalación",
    "fotos_instalacion": "fotos digitales del interior y exterior de la instalación",
    "exclusion_categorica": (
        "determinación de cumplimiento ambiental: certificación de exclusión categórica"
    ),
    "antecedentes_penales": "certificado de antecedentes penales",
    "no_deuda_asume": "certificado de no deuda en pensiones alimentarias (ASUME)",
    "no_deuda_crim": "certificación de no deuda del CRIM (propiedad mueble)",
    "planillas_ingresos": "certificación de radicación de planillas de ingresos (5 años)",
    "planillas_ivu": "certificación de radicación de planillas del IVU",
    "certificacion_deuda_dh": "certificación de deuda del Departamento de Hacienda",
    "certificado_incorporacion": "certificado de incorporación (Departamento de Estado)",
    "numero_patronal": "evidencia del número de identificación patronal o seguro social",
    "registro_comerciantes": "certificación vigente del Registro de Comerciantes",
    "patente_municipal": "copia de la patente municipal o evidencia de pago",
    "proyeccion_ventas": "proyección de ventas anual",
    "sanitaria": "licencia sanitaria",
    "prevencion_incendios": "certificado de prevención de incendios",
    "bebidas_alcoholicas": "licencia de bebidas alcohólicas",
    "ministerial": "uso permitido ministerialmente en el distrito",
    "variacion_aprobada": "variación en el uso previamente autorizada",
    "no_permitido": "uso no permitido en el distrito y sin variación autorizada",
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
    owner_authorization: bool | None = Field(
        default=None,
        description="For a lessee or option-holder: the owner authorises the action (2.1.9.5(b)).",
    )
    municipio: str = Field(min_length=1)
    catastro: str = Field(pattern=r"^\d{3}-\d{3}-\d{3}-\d{2}$")
    catastro_matches_ogpe: bool | None = Field(
        default=None, description="Whether the OGPe catastral records match the Applicant's."
    )
    calificacion: str = Field(min_length=1, description="Zoning district.")

    def core_fields_es(self) -> dict[str, str]:
        standing = LABELS[self.standing_document] if self.standing_document else "ninguno"
        role = LABELS[self.applicant_role]
        if self.owner_authorization is not None:
            authorized = "autorizado por el" if self.owner_authorization else "sin autorización del"
            role += f" ({authorized} dueño para la acción propuesta)"
        catastro = f"{self.municipio}, {self.catastro}"
        if self.catastro_matches_ogpe is not None:
            matches = "concuerda" if self.catastro_matches_ogpe else "no concuerda"
            catastro += f" (la información catastral {matches} con el sistema de la OGPe)"
        return {
            "Solicitante": f"{self.applicant_name} ({LABELS[self.applicant_type]})",
            "Rol del solicitante": role,
            "Documento de legitimación activa": standing,
            "Municipio y número de catastro": catastro,
            "Calificación del distrito": self.calificacion,
        }


class PcocPermitRequest(PermitRequestCore):
    """Permiso de Construcción."""

    type_name_es: ClassVar[str] = "Permiso de Construcción"
    application_kind: ClassVar[ApplicationKind] = "nueva"

    request_type: Literal["PCOC"]
    attached_documents: list[PcocDocument]
    work_description: str = Field(min_length=1)
    estimated_work_value: Decimal | None = Field(default=None, gt=0)
    work_value_method: WorkValueMethod | None = None
    floors: int = Field(ge=1)
    height_m: Decimal = Field(gt=0)
    gross_floor_area_m2: Decimal = Field(gt=0)
    declared_fee: Decimal = Field(ge=0)
    works_by_government: bool = Field(
        default=False,
        description="The works are carried out wholly by the Government (COBROS TERCERO.3); "
        "read only by the Fee Check, for a government Applicant.",
    )
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


class SiteSketch(BaseModel):
    """The croquis of RC-2023-E Sección 3.7.1.2(b)(3) and what it shows."""

    model_config = ConfigDict(extra="forbid")

    floor_area_m2: Decimal = Field(gt=0)
    shows_internal_layout: bool
    shows_access_road: bool
    shows_parking: bool
    shows_dimensions: bool

    def summary_es(self) -> str:
        shown = {
            "la distribución interna": self.shows_internal_layout,
            "la ubicación con respecto a la vía de acceso": self.shows_access_road,
            "el estacionamiento a proveer": self.shows_parking,
            "las dimensiones": self.shows_dimensions,
        }
        return "; ".join(
            [f"área del local {_number(self.floor_area_m2)} m²"]
            + [f"{'incluye' if yes else 'no incluye'} {item}" for item, yes in shown.items()]
        )


class Renewal(BaseModel):
    """The Permiso Único being renewed, per RC-2023-E Sección 3.7.1.4."""

    model_config = ConfigDict(extra="forbid")

    permit_number: str = Field(min_length=1)
    expiry_date: date
    filing_date: date
    use_conforms_certification: bool = Field(
        description="The owner's or operator's certification that the use conforms (3.7.1.4(k))."
    )

    def summary_es(self) -> str:
        days = (self.filing_date - self.expiry_date).days
        if days <= 0:
            timing = f"{-days} días antes de la expiración"
        else:
            timing = f"{days} días ({days / 365:.1f} años) después de la expiración"
        certification = "se acompaña" if self.use_conforms_certification else "no se acompaña"
        return (
            f"renovación del Permiso Único {self.permit_number}, que expira el "
            f"{self.expiry_date.isoformat()}; presentada el {self.filing_date.isoformat()}, "
            f"{timing}; {certification} la certificación del dueño u operador de que el uso "
            "está conforme al Permiso Único"
        )


class PermisoUnicoPermitRequest(PermitRequestCore):
    """Permiso Único: a new request, or the renewal of one already issued."""

    type_name_es: ClassVar[str] = "Permiso Único"

    request_type: Literal["PERMISO_UNICO"]
    attached_documents: list[PermisoUnicoDocument]
    proposed_use: str = Field(min_length=1)
    use_authorization: UseAuthorization
    employees: int = Field(ge=0)
    parking_spaces: int = Field(ge=0)
    required_parking_spaces: int | None = Field(default=None, ge=0)
    operating_hours: str = Field(min_length=1)
    licenses_requested: list[License]
    validity_years: int = Field(ge=1, le=3, description="RC-2023-E Sección 3.7.1.3(a).")
    declared_fee: Decimal = Field(ge=0)
    establishment_type: EstablishmentType | None = Field(
        default=None,
        description="Sets the Licencia Sanitaria rate (COBROS Anejo A, 30); read only by the "
        "Fee Check.",
    )
    site_sketch: SiteSketch | None = None
    in_flood_zone: bool | None = Field(default=None, description="RC-2023-E Sección 3.7.1.1(k).")
    renewal: Renewal | None = None

    @property
    def application_kind(self) -> ApplicationKind:
        return "nueva" if self.renewal is None else "renovacion"

    def fields_es(self) -> dict[str, str]:
        """The request as labelled Spanish fields, as shown to models and Reviewers."""
        kind = {"nueva": "solicitud nueva", "renovacion": "renovación"}[self.application_kind]
        licenses = [LABELS[license] for license in self.licenses_requested] or ["ninguna"]
        documents = "; ".join(LABELS[d] for d in self.attached_documents) or "ninguno"
        parking = f"{self.parking_spaces} espacios de estacionamiento"
        if self.required_parking_spaces is not None:
            parking += f" (requeridos por el distrito {self.required_parking_spaces})"
        if self.in_flood_zone is None:
            flood_zone = "no declarada"
        else:
            flood_zone = f"el predio {'' if self.in_flood_zone else 'no '}ubica en zona inundable"
        fields = {
            "Tipo de solicitud": f"Permiso Único ({kind})",
            **self.core_fields_es(),
            "Uso propuesto": f"{self.proposed_use} ({LABELS[self.use_authorization]})",
            "Empleados / estacionamiento / horario": (
                f"{self.employees} empleados; {parking}; {self.operating_hours}"
            ),
            "Licencias solicitadas": "; ".join(
                licenses + [f"vigencia solicitada {self.validity_years} años"]
            ),
            "Croquis de la instalación": (
                self.site_sketch.summary_es() if self.site_sketch else "no declarado"
            ),
            "Zona inundable": flood_zone,
            "Documentos adjuntos": documents,
            "Derechos declarados": f"${self.declared_fee:,}",
        }
        if self.renewal:
            fields["Renovación"] = self.renewal.summary_es()
        return fields


PermitRequest = Annotated[
    PcocPermitRequest | PermisoUnicoPermitRequest, Field(discriminator="request_type")
]
_PERMIT_REQUEST: TypeAdapter[PermitRequest] = TypeAdapter(PermitRequest)


def parse_permit_request(raw: Mapping[str, Any]) -> PcocPermitRequest | PermisoUnicoPermitRequest:
    """Validate raw fields as the Permit Request their `request_type` names."""
    return _PERMIT_REQUEST.validate_python(raw)
