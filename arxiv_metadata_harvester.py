#!/usr/bin/env python3
import argparse
import os
import time
import json
import csv
from sickle import Sickle
from sickle.models import Record
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Harvest arXiv metadata (id, title, abstract) via OAI-PMH and save to file."
    )
    parser.add_argument(
        "--from-date", "-f", type=str, default=None,
        help="Start date (YYYY-MM-DD) or datestamp to harvest from (inclusive)."
    )
    parser.add_argument(
        "--until-date", "-u", type=str, default=None,
        help="End date (YYYY-MM-DD) or datestamp to harvest until (inclusive)."
    )
    parser.add_argument(
        "--metadata-prefix", "-m", type=str, default="arXiv",
        choices=["arXiv", "arXivRaw"],
        help="Which metadata prefix to use (arXiv gives basic metadata; arXivRaw gives richer version history)."
    )
    parser.add_argument(
        "--set", "-s", type=str, default=None,
        help="OAI set spec to harvest (for example: 'hep-ex')."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="arxiv_metadata.jsonl",
        help="Output file path. Use .jsonl or .csv extension (decides the format)."
    )
    parser.add_argument(
        "--batch-delay", type=float, default=0.1,
        help="Delay (in seconds) between successive record fetches to be polite."
    )
    parser.add_argument(
        "--download-pdfs", action="store_true",
        help="Also download PDF files for each harvested arXiv id into a `pdfs/` subdirectory by default."
    )
    parser.add_argument(
        "--pdf-dir", type=str, default="pdfs",
        help="Directory to save downloaded PDFs when --download-pdfs is set."
    )
    parser.add_argument(
        "--max-records", type=int, default=0,
        help="If >0, stop after this many records (useful for testing)."
    )
    parser.add_argument(
        "--chunk-days", type=int, default=30,
        help="If >0 and both --from-date and --until-date are supplied, split the range into chunks of this many days to avoid oversized OAI responses."
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=None,
        help="Path to checkpoint file to resume (stores last datestamp or resumption token)."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite output file instead of appending."
    )
    return parser.parse_args()

def write_jsonl_line(fout, obj):
    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_csv_row(csv_writer, obj):
    # CSV columns: id, title, abstract, authors, categories, datestamp, sets, doi, comments, journal_ref
    csv_writer.writerow({
        "id": obj.get("id", ""),
        "title": obj.get("title", ""),
        "abstract": obj.get("abstract", ""),
        "authors": ", ".join(obj.get("authors", []) or []),
        "categories": ", ".join(obj.get("categories", []) or []),
        "datestamp": obj.get("datestamp", ""),
        "sets": ", ".join(obj.get("sets", []) or []),
        "doi": obj.get("doi", ""),
        "comments": obj.get("comments", ""),
        "journal_ref": obj.get("journal_ref", ""),
    })

def load_checkpoint(cp_path):
    """Load checkpoint (resumption token or from_date) if present."""
    if not cp_path or not os.path.exists(cp_path):
        return None
    with open(cp_path, "r", encoding="utf-8") as f:
        data = f.read().strip()
        if not data:
            return None
        return data

def save_checkpoint(cp_path, token_or_date):
    if not cp_path:
        return
    with open(cp_path, "w", encoding="utf-8") as f:
        f.write(token_or_date or "")

def harvest(
    from_date, until_date,
    metadata_prefix,
    output_path, batch_delay=0.1,
    checkpoint_path=None,
    overwrite=False,
    set_spec=None,
    download_pdfs=False,
    pdf_dir="pdfs",
    max_records: int = 0,
    chunk_days: int = 30,
):
    base_url = "https://oaipmh.arxiv.org/oai"
    sickle = Sickle(base_url)
    # If a set_spec (possibly shorthand like 'hep-ex') is provided, try to
    # resolve it against available OAI sets on the server so users can pass
    # short names like 'hep-ex' instead of 'physics:hep-ex'. If resolution
    # fails we leave the value unchanged and allow the server to return an
    # error (which will be surfaced to the user).
    if set_spec:
        try:
            available = [getattr(st, 'setSpec', '') for st in sickle.ListSets()]
            if set_spec not in available:
                # try suffix match (e.g. 'physics:hep-ex' endswith ':hep-ex')
                candidates = [s for s in available if s.endswith(':' + set_spec) or s == set_spec or s.endswith(set_spec)]
                if candidates:
                    chosen = candidates[0]
                    print(f"Resolving set '{set_spec}' -> '{chosen}'")
                    set_spec = chosen
                else:
                    # no match found; warn and continue
                    print(f"Warning: could not resolve set '{set_spec}' against available sets; proceeding with given value")
        except Exception as e:
            print(f"Warning: failed to list sets for resolution: {e}")
    
    # Determine file mode
    file_mode = "w" if overwrite else "a"
    is_csv = output_path.lower().endswith(".csv")
    
    # Possibly resume checkpoint
    resume_token = None
    last_from_date = from_date
    if checkpoint_path:
        cp = load_checkpoint(checkpoint_path)
        if cp:
            # If it looks like a token (starts with something), prefer resumption
            resume_token = cp
            print(f"Resuming from checkpoint token: {resume_token}")
    
    # Open output
    fout = open(output_path, file_mode, encoding="utf-8")
    csv_writer = None
    if is_csv:
        fieldnames = ["id", "title", "abstract"]
        csv_writer = csv.DictWriter(fout, fieldnames=fieldnames)
        # Write header if we're overwriting, or if appending to a newly created/empty file
        try:
            should_write_header = (file_mode == "w")
            if not should_write_header and file_mode == "a":
                try:
                    fout.seek(0, os.SEEK_END)
                    if fout.tell() == 0:
                        should_write_header = True
                except Exception:
                    should_write_header = True
            if should_write_header:
                csv_writer.writeheader()
        except Exception:
            # best-effort: if header write fails we continue
            pass
    
    def _list_records_for_params(p):
        try:
            return sickle.ListRecords(**p)
        except Exception:
            # re-raise to be handled by caller
            raise

    # Helper to iterate over date chunks if requested. This avoids a single
    # very large OAI response that some servers reject. If chunk_days <= 0
    # we do a single request for the whole range.
    def date_chunks(start: str, end: str, days: int):
        from datetime import timedelta
        fmt = "%Y-%m-%d"
        s = datetime.strptime(start, fmt)
        e = datetime.strptime(end, fmt)
        if s > e:
            return []
        if days <= 0:
            return [(start, end)]
        chunks = []
        cur = s
        while cur <= e:
            nxt = cur + timedelta(days=days - 1)
            if nxt > e:
                nxt = e
            chunks.append((cur.strftime(fmt), nxt.strftime(fmt)))
            cur = nxt + timedelta(days=1)
        return chunks

    count = 0
    params_base = {"metadataPrefix": metadata_prefix}
    if set_spec:
        params_base["set"] = set_spec
    pdf_dir_created = False
    try:
        # If both from and until dates are provided and chunking enabled,
        # iterate per-chunk. Otherwise do a single ListRecords call.
        if last_from_date and until_date and chunk_days and chunk_days > 0:
            chunks = date_chunks(last_from_date, until_date, chunk_days)
        else:
            chunks = [(last_from_date, until_date)]

        for ch_from, ch_until in chunks:
            params = dict(params_base)
            if ch_from:
                params["from"] = ch_from
            if ch_until:
                params["until"] = ch_until
            if resume_token:
                params = {"resumptionToken": resume_token}

            print("Starting harvest with params:", params)
            records = _list_records_for_params(params)
            for rec in records:
                rec_id = rec.header.identifier
                md = rec.metadata
                title = None
                abstract = None
                if "title" in md:
                    t = md["title"]
                    if isinstance(t, list):
                        title = " ".join(t)
                    else:
                        title = t
                if "abstract" in md:
                    a = md["abstract"]
                    if isinstance(a, list):
                        abstract = " ".join(a)
                    else:
                        abstract = a
                # extract common fields if available in metadata
                # Attempt to normalize author names. Sickle/OAI metadata is
                # inconsistent: some records have 'author' or 'authors' as
                # strings/lists, while others contain name parts like
                # 'forenames' / 'keyname'. We'll try several patterns and
                # create readable "Firstname Lastname" strings.
                raw_authors = md.get("creator") or md.get("authors") or md.get("author") or []
                if isinstance(raw_authors, str):
                    raw_authors = [raw_authors]

                norm_authors = []
                name_parts_list = []  # preserve original name parts per-author
                # If the metadata supplies separate author name arrays (keyname/forenames)
                # try to combine them.
                keynames = md.get("keyname")
                forenames = md.get("forenames")
                if keynames and forenames and isinstance(keynames, list) and isinstance(forenames, list):
                    for k, f in zip(keynames, forenames):
                        name_str = f"{f} {k}".strip()
                        norm_authors.append(name_str)
                        name_parts_list.append({"keyname": k, "forenames": f})
                else:
                    # Fallback: normalize strings or other structures
                    for a in raw_authors:
                        if isinstance(a, dict):
                            # try common dict keys
                            kn = a.get("keyname") or a.get("family") or a.get("last")
                            fn = a.get("forenames") or a.get("given") or a.get("first")
                            if kn or fn:
                                name_str = (f"{fn} {kn}" if fn else kn).strip()
                                norm_authors.append(name_str)
                                name_parts_list.append({"keyname": kn, "forenames": fn})
                            else:
                                s = str(a)
                                norm_authors.append(s)
                                name_parts_list.append({"raw": a})
                        else:
                            s = str(a)
                            norm_authors.append(s)
                            name_parts_list.append({"raw": a})
                authors = norm_authors
                categories = md.get("categories") or md.get("subject") or md.get("category") or []
                if isinstance(categories, str):
                    categories = [categories]
                datestamp = getattr(rec.header, "datestamp", None) or md.get("date")
                sets = []
                try:
                    sets = list(rec.header.setSpecs)
                except Exception:
                    # Not always present
                    sets = []

                doi = md.get("doi") or md.get("identifier")
                comments = md.get("comment") or md.get("comments")
                journal_ref = md.get("journal-ref") or md.get("journal_ref") or md.get("journal")

                # Build a trimmed copy of raw metadata: remove fields we've
                # promoted to top-level to avoid duplication. Keep lower-level
                # author name parts etc. in the raw metadata.
                raw_md = dict(md) if isinstance(md, dict) else {}
                promoted_keys = [
                    "id",
                    "title",
                    "abstract",
                    "authors",
                    "author",
                    "creator",
                    "categories",
                    "category",
                    "subject",
                    "doi",
                    "comments",
                    "comment",
                    "journal-ref",
                    "journal_ref",
                    "journal",
                    "created",
                    "updated",
                ]
                for k in promoted_keys:
                    raw_md.pop(k, None)
                # keep the original author name parts under raw_metadata['name_parts']
                if name_parts_list:
                    raw_md["name_parts"] = name_parts_list

                obj = {
                    "id": rec_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "categories": categories,
                    "datestamp": datestamp,
                    "sets": sets,
                    "doi": doi,
                    "comments": comments,
                    "journal_ref": journal_ref,
                    "raw_metadata": raw_md,
                }

                if is_csv:
                    write_csv_row(csv_writer, obj)
                else:
                    write_jsonl_line(fout, obj)

                count += 1
                if download_pdfs:
                    # extract arXiv id (strip any oai: prefix)
                    try:
                        arxiv_id = str(rec_id).split(":")[-1]
                        # create pdf dir lazily
                        if not pdf_dir_created:
                            os.makedirs(pdf_dir, exist_ok=True)
                            pdf_dir_created = True
                        download_pdf(arxiv_id, pdf_dir)
                    except Exception as e:
                        print(f"Failed to download PDF for {rec_id}: {e}")
                if count % 100 == 0:
                    print(f"Fetched {count} records…")

                # Optionally save checkpoint after each record (or every N)
                # Here, the “resumptionToken” is available via rec._resumption_token of the iterator
                # (But Sickle’s ListRecords hides that). For safety, you might also checkpoint by latest datestamp.

                time.sleep(batch_delay)
                if max_records and count >= max_records:
                    print(f"Reached max_records={max_records}; stopping.")
                    break
            # ensure we break out of the outer chunk loop if requested
            if max_records and count >= max_records:
                break
        print(f"Done harvesting. Total records: {count}")
    except Exception as e:
        print("Encountered exception during harvesting:", e)
    finally:
        # Try to save a checkpoint (if available)
        # In Sickle, after iteration, records.resumption_token might hold the last token
        try:
            # Only attempt to access `records` if it exists and has the attribute.
            if 'records' in locals() and hasattr(records, 'resumption_token'):
                last_token = records.resumption_token
                if last_token:
                    print("Saving resumptionToken to checkpoint:", last_token)
                    save_checkpoint(checkpoint_path, last_token)
        except Exception:
            # best-effort: ignore errors while saving checkpoint
            pass

        # Or fallback: save the last from_date (if using that style).
        # Only save this if there was no resumption token saved above.
        try:
            saved_token = None
            if 'records' in locals() and hasattr(records, 'resumption_token'):
                saved_token = records.resumption_token
        except Exception:
            saved_token = None

        if checkpoint_path and not saved_token:
            # try save last_from_date
            if last_from_date:
                save_checkpoint(checkpoint_path, last_from_date)
        fout.close()


def download_pdf(arxiv_id: str, out_dir: str, attempts: int = 3):
    """Download the PDF for a given arXiv id into out_dir.

    This is a best-effort helper: it makes a simple request to
    https://arxiv.org/pdf/{arxiv_id}.pdf and saves it. It retries a few
    times on transient failures.
    """
    import requests

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    out_path = os.path.join(out_dir, f"{arxiv_id.replace('/', '_')}.pdf")
    if os.path.exists(out_path):
        return out_path

    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, stream=True, timeout=30)
            if resp.status_code == 200:
                with open(out_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                return out_path
            else:
                raise RuntimeError(f"HTTP {resp.status_code}")
        except Exception as e:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def main():
    args = parse_args()
    harvest(
        from_date=args.from_date,
        until_date=args.until_date,
        metadata_prefix=args.metadata_prefix,
        output_path=args.output,
        batch_delay=args.batch_delay,
        checkpoint_path=args.checkpoint,
        overwrite=args.overwrite,
        set_spec=args.set,
        download_pdfs=args.download_pdfs,
        pdf_dir=args.pdf_dir,
        max_records=args.max_records,
        chunk_days=args.chunk_days,
    )

if __name__ == "__main__":
    main()