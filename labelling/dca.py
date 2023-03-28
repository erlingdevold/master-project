from dataclasses import dataclass



# T = TypeVar("T")


# def from_int(x: Any) -> int:
#     assert isinstance(x, int) and not isinstance(x, bool)
#     return x


# def from_str(x: Any) -> str:
#     assert isinstance(x, str)
#     return x


# def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
#     assert isinstance(x, list)
#     return [f(y) for y in x]


# def to_class(c: Type[T], x: Any) -> dict:
#     assert isinstance(x, c)
#     return cast(Any, x).to_dict()

@dataclass
class DCA():
    relevant_år: int
    meldingsår: int
    meldingstype_kode: str
    meldingstype: str
    meldingsnummer: int
    meldingsversjon: int
    sekvensnummer: int
    melding_id: int
    meldingstidspunkt: str
    meldingsdato: str
    meldingsklokkeslett: str
    radiokallesignal_ers: str
    fartøynavn_ers: str
    registreringsmerke_ers: str
    fartøynasjonalitet_kode: str
    fartøygruppe_kode: str
    fartøygruppe: str
    kvotetype_kode: int
    kvotetype: str
    aktivitet_kode: str
    aktivitet: str
    havn_kode: str
    havn: str
    havn_nasjonalitet: str
    starttidspunkt: str
    startdato: str
    startklokkeslett: str
    startposisjon_bredde: str
    startposisjon_lengde: str
    hovedområde_start_kode: str
    hovedområde_start: str
    lokasjon_start_kode: str
    sone_kode: str
    sone: str
    områdegruppering_start_kode: str
    områdegruppering_start: str
    havdybde_start: int
    stopptidspunkt: str
    stoppdato: str
    stoppklokkeslett: str
    varighet: int
    fangstår: int
    stopposisjon_bredde: str
    stopposisjon_lengde: str
    hovedområde_stopp_kode: str
    hovedområde_stopp: str
    lokasjon_stopp_kode: str
    områdegruppering_stopp_kode: str
    områdegruppering_stopp: str
    havdybde_stopp: int
    trekkavstand: int
    pumpet_fra_fartøy: str
    redskap_fao_kode: str
    redskap_fao: str
    redskap_fdir_kode: int
    redskap_fdir: str
    redskap_gruppe_kode: int
    redskap_gruppe: str
    redskap_hovedgruppe_kode: int
    redskap_hovedgruppe: str
    redskapsspesifikasjon_kode: int
    redskapsspesifikasjon: str
    redskap_maskevidde: int
    redskap_problem_kode: int
    redskap_problem: str
    redskap_mengde: str
    hovedart_fao_kode: str
    hovedart_fao: str
    hovedart_fdir_kode: int
    art_fao_kode: str
    art_fao: str
    art_fdir_kode: int
    art_fdir: str
    art_gruppe_kode: str
    art_gruppe: str
    art_hovedgruppe_kode: str
    art_hovedgruppe: str
    sildebestand_kode: str
    sildebestand: str
    sildebestand_fdir_kode: str
    rundvekt: int
    individnummer: str
    kjønn_kode: str
    kjønn: str
    lengde: str
    omkrets: str
    spekkmål_a: str
    spekkmål_b: str
    spekkmål_c: str
    fosterlengde: str
    granatnummer: str
    fartøy_id: int
    registreringsmerke: str
    radiokallesignal: str
    fartøynavn: str
    fartøykommune_kode: int
    fartøykommune: str
    fartøyfylke_kode: int
    fartøyfylke: str
    største_lengde: str
    lengdegruppe_kode: int
    lengdegruppe: str
    bruttotonnasje_1969: int
    bruttotonnasje_annen: str
    byggeår: int
    ombyggingsår: int
    motorkraft: int
    motorbyggeår: int
    fartøymateriale_kode: str
    bredde: str
    fartøy_gjelder_fra_dato: str
    fartøy_gjelder_til_dato: str
    fartøyidentifikasjon: str
    fartøylengde: str
