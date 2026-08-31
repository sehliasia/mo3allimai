import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.base import Base
from app.models.cefr_knowledge import CEFRDescriptor, CEFRDescriptorSource, CEFRLevel
from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument
from app.services.cefr_import_service import CEFRImportService
from app.services.cefr_knowledge_service import CEFRKnowledgeService
from app.services.cefr_parser import CEFRParser


def _db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'cefr.db'}")
    Base.metadata.create_all(engine)
    return Session(engine)


def _document(db):
    document = KnowledgeDocument(title="CEFR source", original_filename="cefr.pdf", stored_filename="cefr.pdf", file_path="private", mime_type="application/pdf", file_size=1, uploaded_by=1)
    db.add(document)
    db.commit()
    return document


def _chunk(db, text, *, index=0, quality="partially_structured", page=76, heading="Interaction orale générale"):
    document = db.scalar(select(KnowledgeDocument))
    chunk = KnowledgeChunk(document_id=document.id, chunk_index=index, content=text, content_for_embedding=text, token_count=10, content_type="table", source_page_start=page, source_page_end=page, quality_status="complete", heading_context=[heading], chunk_metadata={"structural_quality": quality}, chunk_hash=f"hash-{index}", ingestion_version="test")
    db.add(chunk)
    db.flush()
    return chunk


def test_parser_splits_multiple_serialized_rows_and_preserves_unicode():
    result = CEFRParser().parse("C2, Scale X = descriptor C2. C1, Scale X = descriptor C1. B2, Scale X = يمكنه إدارة تفاعلات بسيطة.", source_chunk_ids=[1])
    assert [(item.level_code, item.scale_name) for item in result.records] == [("C2", "Scale X"), ("C1", "Scale X"), ("B2", "Scale X")]
    assert result.records[0].descriptor_text == "descriptor C2."
    assert "C1, Scale X" not in result.records[0].descriptor_text
    assert "يمكنه" in result.records[2].descriptor_text


def test_numeric_and_incomplete_scale_fragments_are_ambiguous_not_records():
    parser = CEFRParser()
    numeric = parser.parse("A1, 1 = Peut gérer une interaction.", source_chunk_ids=[1])
    fragment = parser.parse("A1, traitement = Peut gérer une interaction.", source_chunk_ids=[1])
    assert numeric.records == [] and numeric.rejected[0].reason == "NUMERIC_OR_PUNCTUATION_SCALE"
    assert fragment.records == [] and fragment.rejected[0].reason == "INCOMPLETE_SCALE_FRAGMENT"
    assert parser.parse("A1, Écoute = Peut comprendre une consigne.", source_chunk_ids=[1]).records


def test_no_descriptor_variants_are_structured_without_a_fake_descriptor():
    for value in ("Pas de descripteur disponible", "Pas de descripteur disponible.", "Pas de   descripteur disponible .", "Pas de descripteur disponible, voir C1.", "Pas de descripteur disponible, voir C1.."):
        result = CEFRParser().parse(f"A1, Prise de parole = {value}", source_chunk_ids=[1])
        record = result.records[0]
        assert record.status == "NO_DESCRIPTOR_AVAILABLE"
        assert record.descriptor_text is None
    assert result.records[0].reference_level == "C1"


def test_narrative_and_isolated_table_fragment_are_rejected_once():
    parser = CEFRParser()
    narrative = parser.parse("Le niveau A1 est le niveau le plus élémentaire.", source_chunk_ids=[1])
    fragment = parser.parse("A1 | Peut répondre brièvement.", source_chunk_ids=[1])
    commentary = parser.parse("PRE-A1, Commentaires = Des descripteurs pour ce niveau de compétence sont présentés ci-dessous.", source_chunk_ids=[1])
    assert narrative.rejected == []
    assert fragment.rejected[0].reason == "UNSUPPORTED_STRUCTURE"
    assert commentary.rejected[0].reason == "NARRATIVE_FALSE_POSITIVE"


def test_safe_same_page_scale_continuation_is_reconstructed_with_two_sources(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    first = _chunk(db, "A1, Interaction orale", index=0)
    second = _chunk(db, "générale = Peut gérer des interactions simples.", index=1)
    report = CEFRImportService().import_document(db, document_id=document.id, dry_run=False)
    db.commit()
    descriptor = db.scalar(select(CEFRDescriptor))
    sources = list(db.scalars(select(CEFRDescriptorSource)))
    assert report.parsed_available == 1
    assert descriptor.scale.name == "Interaction orale générale"
    assert {source.chunk_id for source in sources} == {first.id, second.id}


def test_safe_descriptor_continuation_reconstructs_only_adjacent_same_page_chunks(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    first = _chunk(db, "A2, Interaction orale générale = Peut comprendre,", index=0)
    second = _chunk(db, "exemple, information personnelle et familiale.", index=1)
    report = CEFRImportService().import_document(db, document_id=document.id, dry_run=True)
    assert report.parsed_available == 1
    assert report.parsed_examples[0]["chunk_ids"] == [first.id, second.id]
    assert report.parsed_examples[0]["descriptor_preview"].startswith("Peut comprendre, exemple")


def test_unsafe_cross_chunk_scale_reconstruction_is_rejected(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    _chunk(db, "A1, Coopération à visée fonctionnelle", index=0)
    _chunk(db, "Discuter d'un document, = Peut échanger.", index=1)
    report = CEFRImportService().import_document(db, document_id=document.id, dry_run=True)
    assert report.parsed_available == 0
    assert report.unsupported_structure == 1
    assert report.ambiguous == 0


def test_unexplained_mid_sentence_fragment_is_rejected_in_a_scale_context(tmp_path):
    db = _db(tmp_path)
    _document(db)
    _chunk(db, "exemple, information personnelle et familiale.", heading="Interaction orale générale")
    report = CEFRImportService().import_document(db, document_id=1, dry_run=True)
    assert report.parsed_available == 0
    assert report.mid_sentence_truncated == 1
    assert report.truncated == 1


def test_dry_run_has_provenance_and_never_writes_rows(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    _chunk(db, "A1, Interaction orale générale = Peut gérer des interactions simples.", index=0, page=76)
    report = CEFRImportService().import_document(db, document_id=document.id, dry_run=True)
    assert report.parsed_available == 1
    assert report.descriptors_without_source == 0
    assert report.parsed_examples[0]["document_id"] == document.id
    assert report.parsed_examples[0]["page_start"] == 76
    assert report.parsed_examples[0]["chunk_ids"]
    assert db.query(CEFRDescriptor).count() == 0


def test_import_is_idempotent_keeps_sources_and_exact_level_query(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    first = _chunk(db, "A1, Prise de parole = Pas de descripteur disponible.", index=0, page=94)
    second = _chunk(db, "A1, Interaction orale générale = Peut gérer des interactions simples.", index=1, page=76)
    third = _chunk(db, "A1, Interaction orale générale = Peut gérer des interactions simples.", index=2, page=77)
    service = CEFRImportService()
    report = service.import_document(db, document_id=document.id, dry_run=False)
    db.commit()
    assert report.persisted == 2
    assert report.no_descriptor_available == 1
    assert {source.chunk_id for source in db.scalars(select(CEFRDescriptorSource))} == {first.id, second.id, third.id}
    repeat = service.import_document(db, document_id=document.id, dry_run=False)
    db.commit()
    assert repeat.persisted == 0 and repeat.duplicates == 3
    records = CEFRKnowledgeService().get_descriptors(db, level_code="A1", scale_name="Interaction orale générale")
    assert len(records) == 1
    assert CEFRKnowledgeService().get_descriptors(db, level_code="A2") == []


def test_layout_unreliable_is_rejected_without_affecting_other_rows(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    _chunk(db, "A1, تفاعل شفهي = يمكنه إدارة تفاعلات بسيطة.", index=0)
    _chunk(db, "A2, Interaction orale = ignored", index=1, quality="layout_unreliable")
    report = CEFRImportService().import_document(db, document_id=document.id, dry_run=True)
    assert report.parsed_available == 1
    assert report.quality_rejected == 1


def test_editorial_commentary_and_contaminated_scale_are_never_available():
    parser = CEFRParser()
    editorial = parser.parse("C2, Commentaires = Plusieurs modifications présentées dans la liste de l'annexe 7 sont expliquées ici.", source_chunk_ids=[1])
    contaminated = parser.parse("C2, B1.2) a été renforcée. Consulter l'annexe 1. Phonologie, Commentaires = Peut comprendre une consigne.", source_chunk_ids=[1])
    valid = parser.parse("C2, Compréhension générale de l'oral = Peut comprendre une intervention longue.", source_chunk_ids=[1])
    assert editorial.records == [] and editorial.rejected[0].reason == "NARRATIVE_FALSE_POSITIVE"
    assert contaminated.records == [] and contaminated.rejected[0].reason == "CONTAMINATED_SCALE"
    assert len(valid.records) == 1


def test_bare_row_start_cannot_be_completed_by_a_mid_sentence_fragment(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    _chunk(db, "A2, Compréhension générale de l'oral =", index=0)
    _chunk(db, "exemple, information personnelle et familiale de base.", index=1)
    report = CEFRImportService().import_document(db, document_id=document.id, dry_run=True)
    assert report.parsed_available == 0
    assert report.truncated == 2


def test_clean_dry_run_has_zero_available_integrity_failures_and_exact_candidate_accounting(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    _chunk(db, "A1, Interaction orale générale = Peut gérer des échanges simples.", index=0)
    _chunk(db, "A2, Interaction orale générale = Pas de descripteur disponible, voir C1..", index=1)
    report = CEFRImportService().import_document(db, document_id=document.id, dry_run=True)
    assert report.available_integrity_failures == 0
    assert report.available_suspicious_scales == 0
    assert report.available_no_descriptor_misclassified == 0
    assert report.available_mid_sentence == 0
    assert report.available_narrative == 0
    assert report.candidate_rows_detected == report.parsed_available + report.no_descriptor_available + report.ambiguous + report.truncated + report.unsupported_structure
    assert db.query(CEFRDescriptor).count() == 0


def test_available_record_requires_source_provenance():
    result = CEFRParser().parse("A1, Interaction orale générale = Peut gérer des échanges simples.", source_chunk_ids=[])
    assert result.records == []
    assert result.rejected[0].reason == "MISSING_PROVENANCE"


def test_complete_parenthetical_long_scale_is_valid_but_incomplete_or_narrative_scales_are_rejected():
    parser = CEFRParser()
    long_scale = "Coopération à visée fonctionnelle (faire la cuisine ensemble, discuter d'un document, organiser un événement, etc.)"
    valid = parser.parse(f"A2, {long_scale} = Peut coopérer avec des interlocuteurs.", source_chunk_ids=[1])
    incomplete = parser.parse("A2, Coopération à visée fonctionnelle (faire la cuisine ensemble, discuter d'un document, = Peut coopérer.", source_chunk_ids=[1])
    narrative = parser.parse("A2, Coopération à visée fonctionnelle. Cette section présente les changements = Peut coopérer.", source_chunk_ids=[1])
    assert valid.records[0].scale_name == long_scale
    assert incomplete.rejected[0].reason == "INCOMPLETE_SCALE_FRAGMENT"
    assert narrative.rejected[0].reason == "CONTAMINATED_SCALE"


def test_descriptor_tail_contamination_is_rejected_without_rejecting_normal_commas():
    parser = CEFRParser()
    contaminated = parser.parse("B2, Comprendre des médias audio ou signés et des enregistrements = Peut comprendre le contenu des informations.. , Comprendre des méd...", source_chunk_ids=[1])
    normal = parser.parse("B2, Comprendre des médias audio ou signés et des enregistrements = Peut comprendre, avec précision, le contenu des informations.", source_chunk_ids=[1])
    assert contaminated.records == [] and contaminated.rejected[0].reason == "TAIL_CONTAMINATION"
    assert normal.records[0].descriptor_text == "Peut comprendre, avec précision, le contenu des informations."


def test_dry_run_reports_long_scale_recovery_tail_and_incomplete_fragment_counters(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    _chunk(db, "A2, Coopération à visée fonctionnelle (faire la cuisine ensemble, discuter d'un document, organiser un événement, etc.) = Peut coopérer.", index=0)
    _chunk(db, "B2, Comprendre des médias = Peut comprendre le contenu des informations.. , Comprendre des méd...", index=1)
    _chunk(db, "B1, Coopération (faire la cuisine, = Peut coopérer.", index=2)
    report = CEFRImportService().import_document(db, document_id=document.id, dry_run=True)
    assert report.valid_long_scales_recovered == 1
    assert report.tail_contamination_rejected == 1
    assert report.available_tail_contamination == 0
    assert report.incomplete_scale_fragments_rejected == 1
    assert db.query(CEFRDescriptor).count() == 0


def test_leading_fragments_are_not_independent_available_records():
    parser = CEFRParser()
    lowercase = parser.parse("B1, Comprendre des médias = enregistrés, sur des sujets familiers.", source_chunk_ids=[1])
    closing_parenthesis = parser.parse("B1, Comprendre des médias = de vacances), à condition que le débit soit lent.", source_chunk_ids=[1])
    assert lowercase.records == [] and lowercase.rejected[0].reason == "AVAILABLE_LEADING_FRAGMENT"
    assert closing_parenthesis.records == [] and closing_parenthesis.rejected[0].reason == "AVAILABLE_LEADING_FRAGMENT"


def test_same_row_fragment_headers_reconstruct_only_a_complete_descriptor():
    parser = CEFRParser()
    result = parser.parse(
        "B1, Comprendre des médias = Peut comprendre les points principaux d'histoires ou de récits (par exemple récit. "
        "B1, Comprendre des médias = de vacances), à condition que le débit soit lent et clair.",
        source_chunk_ids=[1],
    )
    assert len(result.records) == 1
    assert result.records[0].reconstructed_from_fragments is True
    assert "récit. de vacances)" in result.records[0].descriptor_text


def test_cross_chunk_same_row_reconstruction_keeps_all_provenance(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    first = _chunk(db, "B1, Comprendre des médias = Peut comprendre des récits (par exemple récit.", index=0)
    second = _chunk(db, "B1, Comprendre des médias = de vacances), à condition que le débit soit lent.", index=1)
    report = CEFRImportService().import_document(db, document_id=document.id, dry_run=True)
    assert report.parsed_available == 1
    assert report.leading_fragment_reconstructed == 1
    assert report.parsed_examples[0]["chunk_ids"] == [first.id, second.id]


def test_complete_same_level_scale_descriptors_and_nonstandard_starts_remain_valid():
    parser = CEFRParser()
    complete = parser.parse("B1, Comprendre des médias = Peut comprendre un récit simple. B1, Comprendre des médias = Peut identifier les informations principales.", source_chunk_ids=[1])
    quoted = parser.parse('B1, Comprendre des médias = "Info trafic" : peut comprendre les informations principales.', source_chunk_ids=[1])
    numbered = parser.parse("B1, Comprendre des médias = 24 heures sur 24, peut identifier les informations principales.", source_chunk_ids=[1])
    assert len(complete.records) == 2
    assert quoted.records and numbered.records


def test_same_descriptor_and_same_chunk_repeated_is_one_source_link(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    chunk = _chunk(db, "A1, Interaction orale générale = Peut gérer des échanges simples. A1, Interaction orale générale = Peut gérer des échanges simples.")
    report = CEFRImportService().import_document(db, document_id=document.id, dry_run=False)
    db.commit()
    links = list(db.scalars(select(CEFRDescriptorSource)))
    assert db.query(CEFRDescriptor).count() == 1
    assert [(link.chunk_id, link.source_order) for link in links] == [(chunk.id, chunk.chunk_index)]
    assert (report.source_links_seen, report.source_links_unique, report.source_links_deduplicated, report.source_links_inserted) == (2, 1, 1, 1)


def test_same_descriptor_keeps_distinct_sources_and_second_apply_reuses_them(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    first = _chunk(db, "A1, Interaction orale générale = Peut gérer des échanges simples.", index=0)
    second = _chunk(db, "A1, Interaction orale générale = Peut gérer des échanges simples.", index=1)
    service = CEFRImportService()
    initial = service.import_document(db, document_id=document.id, dry_run=False)
    db.commit()
    assert [(link.chunk_id, link.source_order) for link in db.scalars(select(CEFRDescriptorSource).order_by(CEFRDescriptorSource.source_order))] == [(first.id, 0), (second.id, 1)]
    assert initial.source_links_inserted == 2
    repeat = service.import_document(db, document_id=document.id, dry_run=False)
    db.commit()
    assert db.query(CEFRDescriptorSource).count() == 2
    assert repeat.source_links_existing == 2
    assert repeat.source_links_inserted == 0


def test_source_constraint_remains_and_failed_import_rolls_back_document_changes(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    _chunk(db, "A1, Interaction orale générale = Peut gérer des échanges simples.")
    service = CEFRImportService()

    def fail_scale(_db, _name):
        raise RuntimeError("forced import failure")

    service._scale = fail_scale
    with pytest.raises(RuntimeError, match="forced import failure"):
        service.import_document(db, document_id=document.id, dry_run=False)
    db.rollback()
    assert db.query(CEFRLevel).count() == 0
    assert db.query(CEFRDescriptor).count() == 0
    assert db.query(CEFRDescriptorSource).count() == 0
    assert "uq_cefr_descriptor_source_chunk" in {constraint.name for constraint in CEFRDescriptorSource.__table__.constraints}


def test_serialized_cell_is_rejected_and_reference_level_is_persisted(tmp_path):
    parser = CEFRParser()
    contaminated = parser.parse("A1, Interaction orale générale = Peut échanger simplement.. , Interaction orale générale = Peut poursuivre l'échange.", source_chunk_ids=[1])
    assert contaminated.records == []
    assert contaminated.rejected[0].reason == "EMBEDDED_SERIALIZED_CELL"

    db = _db(tmp_path)
    document = _document(db)
    _chunk(db, "C2, Comprendre des annonces = Pas de descripteur disponible, voir C1.")
    CEFRImportService().import_document(db, document_id=document.id, dry_run=False)
    db.commit()
    descriptor = db.scalar(select(CEFRDescriptor))
    assert descriptor.status == "NO_DESCRIPTOR_AVAILABLE"
    assert descriptor.reference_level.code == "C1"


def test_explicit_document_replacement_keeps_levels_and_rebuilds_only_its_projection(tmp_path):
    db = _db(tmp_path)
    document = _document(db)
    _chunk(db, "A1, Interaction orale générale = Peut gérer des échanges simples.")
    service = CEFRImportService()
    service.import_document(db, document_id=document.id, dry_run=False)
    db.commit()
    replaced = service.replace_document(db, document_id=document.id, dry_run=False)
    db.commit()
    assert replaced.persisted == 1
    assert db.query(CEFRDescriptor).count() == 1
    assert db.query(CEFRDescriptorSource).count() == 1
    assert db.query(CEFRLevel).count() >= 7
