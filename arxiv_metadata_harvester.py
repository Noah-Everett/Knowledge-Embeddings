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
        "--output", "-o", type=str, default="arxiv_metadata.jsonl",
        help="Output file path. Use .jsonl or .csv extension (decides the format)."
    )
    parser.add_argument(
        "--batch-delay", type=float, default=0.1,
        help="Delay (in seconds) between successive record fetches to be polite."
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
    # header: id, title, abstract
    csv_writer.writerow({
        "id": obj.get("id", ""),
        "title": obj.get("title", ""),
        "abstract": obj.get("abstract", "")
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
    overwrite=False
):
    base_url = "https://oaipmh.arxiv.org/oai"
    sickle = Sickle(base_url)
    
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
        if file_mode == "w":
            csv_writer.writeheader()
    
    # Prepare parameters
    params = {"metadataPrefix": metadata_prefix}
    if last_from_date:
        params["from"] = last_from_date
    if until_date:
        params["until"] = until_date
    if resume_token:
        # override and resume
        params = {"resumptionToken": resume_token}
    
    print("Starting harvest with params:", params)
    records = sickle.ListRecords(**params)
    count = 0
    try:
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
            obj = {"id": rec_id, "title": title, "abstract": abstract}
            
            if is_csv:
                write_csv_row(csv_writer, obj)
            else:
                write_jsonl_line(fout, obj)
            
            count += 1
            if count % 100 == 0:
                print(f"Fetched {count} records…")
            
            # Optionally save checkpoint after each record (or every N)
            # Here, the “resumptionToken” is available via rec._resumption_token of the iterator
            # (But Sickle’s ListRecords hides that). For safety, you might also checkpoint by latest datestamp.
            
            time.sleep(batch_delay)
        print(f"Done harvesting. Total records: {count}")
    except Exception as e:
        print("Encountered exception during harvesting:", e)
    finally:
        # Try to save a checkpoint (if available)
        # In Sickle, after iteration, records.resumption_token might hold the last token
        try:
            last_token = records.resumption_token
            if last_token:
                print("Saving resumptionToken to checkpoint:", last_token)
                save_checkpoint(checkpoint_path, last_token)
        except Exception:
            pass
        
        # Or fallback: save the last from_date (if using that style)
        if checkpoint_path and not records.resumption_token:
            # try save last_from_date
            if last_from_date:
                save_checkpoint(checkpoint_path, last_from_date)
        fout.close()

def main():
    args = parse_args()
    harvest(
        from_date=args.from_date,
        until_date=args.until_date,
        metadata_prefix=args.metadata_prefix,
        output_path=args.output,
        batch_delay=args.batch_delay,
        checkpoint_path=args.checkpoint,
        overwrite=args.overwrite
    )

if __name__ == "__main__":
    main()