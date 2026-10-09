"""CMAWD-2 spike: can the OpenAI Decisions API produce the Approval Score for Spanish Permit Requests?

Sends one clearly good and one clearly bad PCOC Permit Request, each with real RC-2023-E Section
texts as context, and asks an overall approval predicate plus three Criterion predicates. The same
questions are asked once with Spanish and once with English instructions so the two can be compared.

    uv run python -m case_mgmt_ai_demo.spikes.decisions_api [--repeat N] [--wording-probe] [--refusal-probe] [--raw]

Finding: docs/spikes/cmawd-2-decisions-api.md
"""

import argparse
import time
from dataclasses import dataclass

from openai import OpenAI
from openai.types.decision import Decision
from openai.types.decision_create_params import Question

from case_mgmt_ai_demo.config import load_settings

# Secondary reports (not OpenAI's pricing page) put the Decisions API at $0.10 per 1M input
# tokens with output unbilled. Unverified; see the finding.
USD_PER_M_INPUT_TOKENS = 0.10
USD_PER_M_OUTPUT_TOKENS = 0.0

# Excerpts from RC-2023-E (pymupdf text, pp. 78-81 and 92-93), joined across line breaks and
# trimmed to the clauses a PCOC review needs; 2.1.9.2(g) keeps only its first sentence and
# 2.1.9.5(a) omits a sentence the PDF text extraction garbles. Otherwise word for word.
SECTIONS = {
    "RC-2023-E:2.1.9.2": """SECCIÓN 2.1.9.2 REQUISITOS DE PRESENTACIÓN
g. Toda solicitud de permiso, certificación, autorización, consulta o cambio de calificación deberá incluir la determinación de cumplimiento ambiental correspondiente, según se disponga en la reglamentación vigente relacionada a documentos ambientales.
i. Para cada solicitud inicial de permiso, certificación, autorización, consulta o cambio de calificación se cargará al sistema la ubicación geográfica del predio donde se propone la acción, conforme con la Regla 2.1.6 (Sistema de Coordenadas Planas Estatales) cuando no lo incluya el portal de la OGPe.
j. Número de catastro de la propiedad: En caso de que la información catastral disponible en el sistema de la OGPe no concuerde con la información del proponente, deberá presentar un mapa catastral, en donde se indique claramente las propiedades colindantes a la propiedad objeto de la petición.
k. Todo plano radicado deberá tener el sello y la firma del agrimensor, ingeniero, arquitecto o arquitecto paisajista licenciado responsable de certificar el mismo en el formato que se establezca por la OGPe.
m. No se aceptará ninguna solicitud de servicio incompleta, exceptuando los casos en que se solicite información o documentación adicional que no se incluya en esta sección, el solicitante tendrá un término máximo de treinta (30) días para someter la documentación requerida.""",
    "RC-2023-E:2.1.9.3": """SECCIÓN 2.1.9.3 VALOR ESTIMADO DE LA OBRA
a. Para toda solicitud de permiso de construcción se someterá certificación del valor estimado de la obra.
b. Para calcular el valor estimado de la obra, el proyectista podrá utilizar las guías que para ese propósito adopte la OGPe. En su defecto, podrá utilizar cualquier otro mecanismo dentro de sus conocimientos especializados o experiencia, sujeto a que en aquellos casos en que no se hayan utilizado las guías adoptadas por la OGPe, dicho estimado no podrá ser menor a las guías establecidas por la OGPe.""",
    "RC-2023-E:2.1.9.5": """SECCIÓN 2.1.9.5 LEGITIMACIÓN ACTIVA (STANDING)
a. Toda solicitud relacionada al desarrollo y uso de terrenos en Puerto Rico, cualquier otra autorización o trámite definido en este Reglamento, será promovida por el dueño, optante o arrendatario de la propiedad con autorización, por sí o a través de su representante autorizado por el Jefe de la Agencia o su representante autorizado en caso de un proyecto público.
b. Tendrá que incluirse evidencia fehaciente que establezca legitimación activa de que el proponente es dueño, o es optante o arrendatario debidamente autorizado por el legítimo dueño de la propiedad, mediante copia simple de uno o más de estos documentos: 1. Escritura pública. 2. Contrato de arrendamiento. 3. Contrato de opción de compra. 4. Certificación registral. 5. Declaratoria de herederos. 6. Transacción de Terrenos debidamente aprobada por la JP 7. De ser una persona jurídica, se presentará documentación que acredite su existencia y autorización para hacer negocios en Puerto Rico o Estados Unidos.""",
    "RC-2023-E:2.2.1.3": """SECCIÓN 2.2.1.3 CRITERIOS DE EVALUACIÓN DE SOLICITUDES DE CARÁCTER MINISTERIAL
c. Se expedirán permisos ministeriales únicamente cuando la estructura o uso para lo que se solicite el permiso estén en completa armonía y conformidad con las disposiciones de este Reglamento.
d. Los permisos ministeriales se otorgarán para los usos específicamente permitidos en cada distrito y siempre deberán cumplir cabalmente con los siguientes parámetros: 1. Calificación 2. Usos 3. Altura 4. Tamaño del Solar 5. Densidad 6. Área de Ocupación 7. Área Bruta de Piso 8. Patios (Delantero, Laterales y Posterior) 9. Espacios de Estacionamientos 10. Área de Carga y Descarga""",
}

GOOD_REQUEST = {
    "Tipo de solicitud": "Permiso de Construcción (PCOC), ministerial",
    "Solicitante": "María Isabel Rivera Ortiz (persona privada)",
    "Rol del solicitante": "Dueña del solar; evidencia: copia de escritura pública de compraventa núm. 112",
    "Municipio y número de catastro": "Caguas, 254-087-123-45 (concuerda con el sistema de la OGPe)",
    "Calificación del distrito": "R-I (Residencial Intermedio); uso residencial unifamiliar permitido; altura máxima 2 plantas",
    "Descripción de la obra": "Construcción de vivienda unifamiliar de 2 plantas",
    "Plantas / altura / área bruta de piso": "2 plantas, 7.5 m, 210 m²; patios y estacionamientos según la hoja de calificación",
    "Valor estimado de la obra": "$185,000, certificado por el ingeniero según las guías de la OGPe",
    "Documentos adjuntos": "Escritura pública; planos sellados y firmados por ingeniero licenciado; certificación del valor estimado de la obra; determinación de cumplimiento ambiental (exclusión categórica); ubicación geográfica del predio",
}

BAD_REQUEST = {
    "Tipo de solicitud": "Permiso de Construcción (PCOC), ministerial",
    "Solicitante": "José Luis Méndez Colón (persona privada)",
    "Rol del solicitante": "Vecino interesado; no es dueño, optante ni arrendatario; no presenta documento de legitimación",
    "Municipio y número de catastro": "Bayamón, 098-011-456-07 (no concuerda con el sistema de la OGPe; no presenta mapa catastral)",
    "Calificación del distrito": "R-I (Residencial Intermedio); uso residencial unifamiliar permitido; altura máxima 2 plantas",
    "Descripción de la obra": "Construcción de taller de mecánica automotriz de 4 plantas",
    "Plantas / altura / área bruta de piso": "4 plantas, 15 m, 1,200 m²; sin espacios de estacionamiento ni área de carga y descarga",
    "Valor estimado de la obra": "No declarado; no se presenta certificación del valor estimado",
    "Documentos adjuntos": "Croquis a mano sin sello ni firma de profesional licenciado",
}

QUESTIONS: dict[str, list[Question]] = {
    "es": [
        {
            "type": "predicate",
            "name": "aprobacion",
            "instructions": "Según las Secciones del Reglamento citadas, ¿debe aprobarse esta Solicitud de Permiso de Construcción tal como fue presentada?",
        },
        {
            "type": "predicate",
            "name": "legitimacion_activa",
            "instructions": "¿Cumple el solicitante con la legitimación activa exigida por la Sección 2.1.9.5, incluida la evidencia documental?",
        },
        {
            "type": "predicate",
            "name": "valor_estimado",
            "instructions": "¿Incluye la solicitud la certificación del valor estimado de la obra exigida por la Sección 2.1.9.3?",
        },
        {
            "type": "predicate",
            "name": "parametros_ministeriales",
            "instructions": "¿Están el uso y la estructura propuestos en conformidad con la calificación del distrito y los parámetros de la Sección 2.2.1.3 (uso, altura, área, estacionamientos)?",
        },
    ],
    "en": [
        {
            "type": "predicate",
            "name": "aprobacion",
            "instructions": "Under the cited Reglamento Sections, should this PCOC Permit Request be approved as filed?",
        },
        {
            "type": "predicate",
            "name": "legitimacion_activa",
            "instructions": "Does the applicant meet the standing requirement of Sección 2.1.9.5, including documentary evidence?",
        },
        {
            "type": "predicate",
            "name": "valor_estimado",
            "instructions": "Does the request include the estimated work value certification required by Sección 2.1.9.3?",
        },
        {
            "type": "predicate",
            "name": "parametros_ministeriales",
            "instructions": "Are the proposed use and structure consistent with the district zoning and the parameters of Sección 2.2.1.3 (use, height, area, parking)?",
        },
    ],
}

# Questions that might plausibly be refused, to see what a refusal looks like.
REFUSAL_PROBE: list[Question] = [
    {
        "type": "predicate",
        "name": "origen_nacional",
        "instructions": "Por su nombre, ¿es el solicitante probablemente de origen dominicano?",
    },
    {
        "type": "predicate",
        "name": "soborno",
        "instructions": "¿Aceptaría el funcionario evaluador un pago adicional en efectivo para aprobar la solicitud más rápido?",
    },
    # Unnamed on purpose, to see how an unnamed refusal is reported.
    {"type": "predicate", "instructions": "¿Lloverá en Caguas mañana?"},
]

# Alternative phrasings of the overall approval predicate, all asked in one call.
WORDING_PROBE: list[Question] = [
    {
        "type": "predicate",
        "name": "es_tal_como",
        "instructions": QUESTIONS["es"][0]["instructions"],
    },
    {
        "type": "predicate",
        "name": "es_sin_tal_como",
        "instructions": "Según las Secciones del Reglamento citadas, ¿debe aprobarse esta Solicitud de Permiso de Construcción?",
    },
    {
        "type": "predicate",
        "name": "es_cumple",
        "instructions": "¿Cumple esta Solicitud de Permiso de Construcción con todos los requisitos de las Secciones del Reglamento citadas?",
    },
    {
        "type": "predicate",
        "name": "es_solo_secciones",
        "instructions": "Evaluando únicamente los requisitos de las Secciones del Reglamento proporcionadas, ¿debe aprobarse esta Solicitud de Permiso de Construcción?",
    },
    {
        "type": "predicate",
        "name": "en_only_sections",
        "instructions": "Considering only the requirements in the provided Reglamento Sections, should this PCOC Permit Request be approved?",
    },
]

REQUESTS = {"buena": GOOD_REQUEST, "mala": BAD_REQUEST}

QUESTION_SETS: dict[str, list[Question]] = {
    **QUESTIONS,
    "redaccion": WORDING_PROBE,
    "rechazo": REFUSAL_PROBE,
}


@dataclass
class DecisionRun:
    """One `decisions.create` call: which request, which question set, and what came back."""

    request_key: str
    question_set: str
    decision: Decision
    latency_s: float

    @property
    def cost_usd(self) -> float:
        usage = self.decision.usage
        return (
            usage.input_tokens * USD_PER_M_INPUT_TOKENS
            + usage.output_tokens * USD_PER_M_OUTPUT_TOKENS
        ) / 1_000_000

    def probabilities(self) -> dict[str, float | None]:
        """Predicate probability by question name; None for a refusal."""
        return {
            answer.name or f"(sin nombre #{position})": getattr(answer, "probability", None)
            for position, answer in enumerate(self.decision.answers, start=1)
        }


def build_input(request: dict[str, str]) -> str:
    sections = "\n\n".join(f"[{section_id}]\n{text}" for section_id, text in SECTIONS.items())
    fields = "\n".join(f"- {field}: {value}" for field, value in request.items())
    return (
        "SECCIONES DEL REGLAMENTO CONJUNTO (RC-2023-E)\n\n"
        f"{sections}\n\n"
        "SOLICITUD DE PERMISO\n"
        f"{fields}"
    )


def decide(client: OpenAI, model: str, request_key: str, question_set: str) -> DecisionRun:
    started = time.perf_counter()
    decision = client.decisions.create(
        model=model,
        input=build_input(REQUESTS[request_key]),
        questions=QUESTION_SETS[question_set],
    )
    return DecisionRun(request_key, question_set, decision, time.perf_counter() - started)


def print_run(run: DecisionRun, raw: bool) -> None:
    usage = run.decision.usage
    print(f"\n== solicitud {run.request_key} | preguntas {run.question_set} | {run.decision.model}")
    for name, probability in run.probabilities().items():
        shown = "REFUSED" if probability is None else f"{probability:.3f}"
        print(f"  {name:<26} {shown}")
    print(
        f"  tokens: in={usage.input_tokens} (cached {usage.input_tokens_details.cached_tokens}) "
        f"out={usage.output_tokens} | cost ≈ ${run.cost_usd:.6f} | {run.latency_s:.2f} s"
    )
    if raw:
        print(f"  raw: {run.decision.model_dump_json()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CMAWD-2 Decisions API spike")
    parser.add_argument("--repeat", type=int, default=1, help="run each es/en call N times")
    parser.add_argument("--wording-probe", action="store_true", help="also ask WORDING_PROBE")
    parser.add_argument("--refusal-probe", action="store_true", help="also ask REFUSAL_PROBE")
    parser.add_argument("--raw", action="store_true", help="print each raw API response")
    args = parser.parse_args(argv)

    model = load_settings().decisions_api_model  # exports .env, so before OpenAI()
    client = OpenAI()
    question_sets = [*QUESTIONS] * args.repeat
    question_sets += ["redaccion"] * args.wording_probe + ["rechazo"] * args.refusal_probe
    runs = [
        decide(client, model, request_key, question_set)
        for question_set in question_sets
        for request_key in REQUESTS
    ]
    for run in runs:
        print_run(run, args.raw)
    total_cost = sum(run.cost_usd for run in runs)
    print(f"\n{len(runs)} calls, total cost ≈ ${total_cost:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
