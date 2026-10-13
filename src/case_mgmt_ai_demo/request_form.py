"""The form a Permit Request is created with in the web UI: its fields for each Request Type,
and the raw fields a submission stands for.

A form field is named by the path of the Permit Request field it fills (`renewal.expiry_date`),
so an intake error lands on the input it is about.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from case_mgmt_ai_demo.chunks import RequestType
from case_mgmt_ai_demo.permit_request import (
    ApplicantRole,
    ApplicantType,
    EstablishmentType,
    License,
    PcocDocument,
    PcocPermitRequest,
    PermisoUnicoDocument,
    PermisoUnicoPermitRequest,
    PlansProfessional,
    StandingDocument,
    UseAuthorization,
    WorkValueMethod,
)

# `choice` is one of `options`; `check` is a box that declares the field true; `checklist` is
# any of `options`.
Kind = Literal["text", "integer", "decimal", "date", "choice", "yes_no", "check", "checklist"]

REQUEST_TYPES_ES: dict[RequestType, str] = {
    "PCOC": PcocPermitRequest.type_name_es,
    "PERMISO_UNICO": PermisoUnicoPermitRequest.type_name_es,
}
# The Request Type the form opens on, and shows when the one submitted is not one.
DEFAULT_REQUEST_TYPE: RequestType = "PCOC"


@dataclass(frozen=True)
class FormField:
    name: str
    label: str
    kind: Kind = "text"
    options: tuple[str, ...] = ()  # values, shown under their `permit_request.LABELS`
    required: bool = True  # within its group, when the group is optional
    hint: str | None = None


@dataclass(frozen=True)
class FieldGroup:
    """Fields shown together. An optional group is one nested field of the Permit Request,
    not declared when every field of the group is left blank."""

    title: str
    fields: tuple[FormField, ...]
    optional: bool = False
    hint: str | None = None


def _choice(name: str, label: str, literal: Any, **rest: Any) -> FormField:
    return FormField(name, label, "choice", options=get_args(literal), **rest)


def _documents(literal: Any) -> FieldGroup:
    return FieldGroup(
        "Documentos adjuntos",
        (FormField("attached_documents", "Documentos adjuntos", "checklist", get_args(literal)),),
        hint="Marque cada documento que acompaña la solicitud.",
    )


_DECLARED_FEE = FormField("declared_fee", "Derechos declarados ($)", "decimal")

_CORE = (
    FieldGroup(
        "Solicitante",
        (
            FormField("applicant_name", "Nombre del solicitante"),
            _choice("applicant_type", "Tipo de solicitante", ApplicantType),
            _choice("applicant_role", "Rol del solicitante", ApplicantRole),
            FormField(
                "owner_authorization",
                "El dueño autoriza la acción propuesta",
                "yes_no",
                required=False,
                hint="Para un arrendatario u optante.",
            ),
            _choice(
                "standing_document",
                "Documento de legitimación activa",
                StandingDocument,
                required=False,
            ),
        ),
    ),
    FieldGroup(
        "Predio",
        (
            FormField("municipio", "Municipio"),
            FormField("catastro", "Número de catastro", hint="Formato 000-000-000-00."),
            FormField(
                "catastro_matches_ogpe",
                "La información catastral concuerda con el sistema de la OGPe",
                "yes_no",
                required=False,
            ),
            FormField(
                "calificacion",
                "Calificación del distrito",
                hint="Distrito de calificación del predio y lo que permite.",
            ),
        ),
    ),
)

_PARAMETERS = "district_parameters"
_PLANS = "plans_certification"
_PCOC = (
    FieldGroup(
        "Obra",
        (
            FormField("work_description", "Descripción de la obra"),
            FormField("floors", "Plantas", "integer"),
            FormField("height_m", "Altura (m)", "decimal"),
            FormField("gross_floor_area_m2", "Área bruta de piso (m²)", "decimal"),
            FormField(
                "estimated_work_value", "Valor estimado de la obra ($)", "decimal", required=False
            ),
            _choice(
                "work_value_method",
                "Método de cálculo del valor",
                WorkValueMethod,
                required=False,
            ),
        ),
    ),
    _documents(PcocDocument),
    FieldGroup(
        "Parámetros del distrito",
        (
            FormField(f"{_PARAMETERS}.lot_area_m2", "Tamaño del solar (m²)", "decimal"),
            FormField(f"{_PARAMETERS}.min_lot_area_m2", "Tamaño mínimo del solar (m²)", "decimal"),
            FormField(f"{_PARAMETERS}.dwelling_units", "Unidades de vivienda", "integer"),
            FormField(f"{_PARAMETERS}.max_dwelling_units", "Máximo de unidades", "integer"),
            FormField(f"{_PARAMETERS}.occupation_pct", "Área de ocupación (%)", "decimal"),
            FormField(f"{_PARAMETERS}.max_occupation_pct", "Ocupación máxima (%)", "decimal"),
            FormField(f"{_PARAMETERS}.yards_m.front", "Patio delantero (m)", "decimal"),
            FormField(f"{_PARAMETERS}.min_yards_m.front", "Patio delantero mínimo (m)", "decimal"),
            FormField(f"{_PARAMETERS}.yards_m.side", "Patios laterales (m)", "decimal"),
            FormField(f"{_PARAMETERS}.min_yards_m.side", "Patios laterales mínimos (m)", "decimal"),
            FormField(f"{_PARAMETERS}.yards_m.rear", "Patio posterior (m)", "decimal"),
            FormField(f"{_PARAMETERS}.min_yards_m.rear", "Patio posterior mínimo (m)", "decimal"),
            FormField(f"{_PARAMETERS}.parking_spaces", "Espacios de estacionamiento", "integer"),
            FormField(
                f"{_PARAMETERS}.required_parking_spaces",
                "Espacios de estacionamiento requeridos",
                "integer",
            ),
            FormField(f"{_PARAMETERS}.loading_spaces", "Espacios de carga y descarga", "integer"),
            FormField(
                f"{_PARAMETERS}.required_loading_spaces",
                "Espacios de carga y descarga requeridos",
                "integer",
            ),
        ),
        optional=True,
        hint="Cada parámetro junto al límite del distrito.",
    ),
    FieldGroup(
        "Certificación de los planos",
        (
            _choice(f"{_PLANS}.professional", "Profesional que los sella y firma", PlansProfessional),
            FormField(f"{_PLANS}.professional_name", "Nombre del profesional"),
            FormField(f"{_PLANS}.license_number", "Número de licencia"),
            FormField(
                f"{_PLANS}.drafted_by",
                "Delineados por",
                required=False,
                hint="Nombre y licencia; en blanco si los delineó quien los certifica.",
            ),
            FormField(
                f"{_PLANS}.sheets_certified",
                "Cada hoja incluye las certificaciones requeridas",
                "yes_no",
            ),
            FormField(
                f"{_PLANS}.compliance_certification",
                "Se acompaña la certificación de cumplimiento del proyectista",
                "yes_no",
            ),
            FormField(f"{_PLANS}.file_format", "Formato del archivo digital", hint="P. ej. DXF."),
            FormField(f"{_PLANS}.north_oriented", "Archivo orientado al norte", "yes_no"),
        ),
        optional=True,
    ),
    FieldGroup(
        "Derechos",
        (
            _DECLARED_FEE,
            FormField(
                "works_by_government",
                "La obra la realiza en su totalidad el Gobierno",
                "check",
                required=False,
            ),
        ),
    ),
)

_PERMISO_UNICO = (
    FieldGroup(
        "Uso",
        (
            FormField("proposed_use", "Uso propuesto"),
            _choice("use_authorization", "Autorización del uso", UseAuthorization),
            FormField("employees", "Empleados", "integer"),
            FormField("parking_spaces", "Espacios de estacionamiento", "integer"),
            FormField(
                "required_parking_spaces",
                "Espacios de estacionamiento requeridos por el distrito",
                "integer",
                required=False,
            ),
            FormField("operating_hours", "Horario de operación"),
            FormField(
                "in_flood_zone", "El predio ubica en zona inundable", "yes_no", required=False
            ),
        ),
    ),
    FieldGroup(
        "Licencias",
        (
            FormField("licenses_requested", "Licencias solicitadas", "checklist", get_args(License)),
            FormField("validity_years", "Años de vigencia solicitados", "integer", hint="De 1 a 3."),
            _choice(
                "establishment_type",
                "Tipo de establecimiento",
                EstablishmentType,
                required=False,
                hint="Fija la tarifa de la Licencia Sanitaria.",
            ),
        ),
    ),
    _documents(PermisoUnicoDocument),
    FieldGroup(
        "Croquis de la instalación",
        (
            FormField("site_sketch.floor_area_m2", "Área del local (m²)", "decimal"),
            FormField("site_sketch.shows_internal_layout", "Muestra la distribución interna", "yes_no"),
            FormField(
                "site_sketch.shows_access_road",
                "Muestra la ubicación con respecto a la vía de acceso",
                "yes_no",
            ),
            FormField("site_sketch.shows_parking", "Muestra el estacionamiento a proveer", "yes_no"),
            FormField("site_sketch.shows_dimensions", "Muestra las dimensiones", "yes_no"),
        ),
        optional=True,
    ),
    FieldGroup(
        "Renovación",
        (
            FormField("renewal.permit_number", "Número del Permiso Único que se renueva"),
            FormField("renewal.expiry_date", "Fecha de expiración", "date"),
            FormField("renewal.filing_date", "Fecha de presentación", "date"),
            FormField(
                "renewal.use_conforms_certification",
                "Se acompaña la certificación de que el uso está conforme",
                "yes_no",
            ),
        ),
        optional=True,
        hint="Solo para renovar un Permiso Único ya expedido.",
    ),
    FieldGroup("Derechos", (_DECLARED_FEE,)),
)

_TYPE_GROUPS: dict[str, tuple[FieldGroup, ...]] = {
    "PCOC": _PCOC,
    "PERMISO_UNICO": _PERMISO_UNICO,
}


def core_groups() -> tuple[FieldGroup, ...]:
    """The fields every Request Type shares."""
    return _CORE


def type_groups(request_type: RequestType) -> tuple[FieldGroup, ...]:
    """The fields of one Request Type, with its required-documents checklist."""
    return _TYPE_GROUPS[request_type]


def errors_by_field(
    request_type: RequestType, errors: Sequence[tuple[str, str]]
) -> tuple[dict[str, str], dict[str, str]]:
    """Intake errors as the form shows them: the first one of each input, by its name, and
    those no input of the form can show, by their path. An error on a nested field as a whole
    (`district_parameters.yards_m` missing) goes to each of its inputs, and one on an item of a
    checklist to the checklist."""
    names = ["request_type"] + [
        field.name for group in (*_CORE, *_TYPE_GROUPS[request_type]) for field in group.fields
    ]
    shown: dict[str, str] = {}
    other: dict[str, str] = {}
    for path, message in errors:
        inputs = [
            name
            for name in names
            if name == path or name.startswith(f"{path}.") or path.startswith(f"{name}.")
        ]
        for name in inputs:
            shown.setdefault(name, message)
        if not inputs:
            other.setdefault(path, message)
    return shown, other


def raw_fields(request_type: str, submitted: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """The raw fields a submitted form stands for, as intake validates them. A field left
    blank is left out; a checklist is the options ticked, none included. A Request Type that
    is not one has no fields of its own, and is left for intake to refuse."""
    raw: dict[str, Any] = {"request_type": request_type}
    for group in (*_CORE, *_TYPE_GROUPS.get(request_type, ())):
        for field in group.fields:
            values = [value.strip() for value in submitted.get(field.name, []) if value.strip()]
            if field.kind != "checklist" and not values:
                continue
            *parents, leaf = field.name.split(".")
            target = raw
            for parent in parents:
                target = target.setdefault(parent, {})
            target[leaf] = values if field.kind == "checklist" else values[0]
    return raw
