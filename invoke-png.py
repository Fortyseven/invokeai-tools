#!/usr/bin/env python3
'''
A swiss army knife for inspecting and renaming Invoke-generated PNG files.
'''

from posixpath import basename
from rich import print
from rich.table import Table
from rich.console import Console
import argparse
import re
import os
import sys
import png
import json

DEFAULT_RENAME_FORMAT_STRING = "{prompt-50}_({seed}, S{steps}, C{cfg_scale})"
rename_rejects = []


class ModeChunk:
    '''
        This dumps all of the tEXt chunks, invoke-created or not.
        There isn't really a need for this, but it's where the
        project started :) Maybe it'll grow into something else
        later.
    '''

    def __init__(self, args, pngfile: str):
        self.args = args
        self.pngfile = pngfile

        self.chunkHandlers = {
            'tEXt': self.renderTextChunk,
        }

    def render(self):
        reader = png.Reader(self.pngfile)
        chunks = reader.chunks()

        for chunk in chunks:
            # check if handlers has a function for this chunk type.
            # i may have over-engineered this.
            if chunk[0].decode() in self.chunkHandlers:
                self.chunkHandlers[chunk[0].decode()](chunk)
            else:
                pass
                # print(f'Unknown chunk type {chunk[0].decode()}')

    def renderTextChunk(self, chunk):
        chunk_subtype, chunk_value = chunk[1].decode().split('\x00')

        # attempt to treat it like json and just pass
        # it through raw if it fails
        try:
            chunk_value = json.loads(chunk_value, strict=False)
            chunk_value = json.dumps(chunk_value)
        except:
            pass

        # use rich to render a table of the text chunks
        table = Table(show_header=True, header_style="bold yellow",
                      title="tEXt", title_justify="left", title_style="bold white")
        table.add_column("subtype", style="dim")
        table.add_column("subtype value")
        table.add_row(chunk_subtype, chunk_value, style="cyan")

        console = Console()
        console.print(table)


class ModeDefault:
    ''' This is the basic "ls"-like listing. '''

    def __init__(self, args, pngfile: str):
        self.args = args
        self.pngfile = pngfile

    def render(self):
        try:
            sd_metadata = getSdMetadataTextChunk(self.pngfile)
            prompt = ""
            if sd_metadata:
                prompt = sd_metadata['image']['prompt'][0]['prompt']
                # let's replace newlines with literal "\n" for display purposes
                prompt = prompt.replace('\n', '\\n')

            width, height = self.getHeaderInfo()

            print(
                f"[b][white]{self.pngfile}[/white][/b], {width}x{height}, '{prompt}'")
        except Exception as e:
            print(
                f"[b][red]{self.pngfile}[/red][/b], parsing error - '{e}'")

    def getHeaderInfo(self):
        reader = png.Reader(self.pngfile)
        png_data = reader.read()
        return list(png_data)[0:2]


class ModeJsonDump:
    ''' This just dumps the raw metadata json from the Invoke-created tEXt chunk. '''

    def __init__(self, args, pngfile: str):
        self.args = args
        self.pngfile = pngfile

    def render(self):
        try:
            if self.args.verbose:
                sd_metadata = getSdMetadataTextChunk(self.pngfile)
                print(json.dumps(sd_metadata, indent=4))
            else:
                sd_metadata = getSdMetadataTextChunk(self.pngfile)['image']
                print(json.dumps(sd_metadata, indent=4))

        except Exception as e:
            print(
                f"[b][red]{self.pngfile}[/red][/b], parsing error - '{e}'")


class ModeRename:
    ''' This performs a bulk rename based on a default, or provided format string. '''

    def __init__(self, args, pngfile: str):
        self.args = args
        self.pngfile = pngfile

    def slugify(self, s):
        s = s.lower().strip()
        s = re.sub(r'[^\w\s-]', '', s)
        s = re.sub(r'[\s_-]+', '-', s)
        s = re.sub(r'^-+|-+$', '', s)
        return s

    def render(self):
        global rename_rejects

        try:
            meta = getSdMetadataTextChunk(self.pngfile)
            meta = meta['image']

            if 'variations' in meta:
                del meta['variations']

            if 'prompt' in meta and 'prompt' in meta['prompt'][0]:
                meta['prompt'] = meta['prompt'][0]['prompt']
                meta['prompt'] = self.slugify(meta['prompt'])

                # Because using a full prompt can lead to very long
                # filenames, we're providing this truncated version
                # of it.
                # TODO: this should be more dynamic, but it's a start (e.g. 'prompt-n' where n is the length of the prompt)
                meta['prompt-50'] = meta['prompt'][0:50]
            else:
                rename_rejects.append(self.pngfile)
                print(
                    f"[b][yellow]{self.pngfile}[/yellow][/b] is an Invoke AI generated PNG file, but it doesn't have a prompt. Skipping.")
                return

            new_basename = self.args.rename_format_string

            # FIXME: if a key doesn't exist, it won't be swapped out
            #        leaving the {key} in the string

            for key in meta:
                new_basename = new_basename.replace(
                    "{"+key+"}", str(meta[key]))

            try:
                new_filename = self.pngfile.replace(
                    basename(self.pngfile), new_basename) + ".png"

                if self.pngfile == new_filename:
                    print(
                        f"[yellow]Skipping {self.pngfile}; already renamed...[/yellow]")
                    return

                if not self.args.dry_run:
                    if (os.path.exists(new_filename)):
                        print(
                            f"[red]Skipping {self.pngfile}; {new_filename} already exists...[/red]")
                        rename_rejects.append(self.pngfile)
                        return
                    os.rename(self.pngfile, new_filename)

                print(
                    f"Renamed [red]{self.pngfile}[/red] to [green]{new_filename}[/green]")
            except Exception as e:
                rename_rejects.append(self.pngfile)
                print(
                    f"[b][red]{self.pngfile}[/red][/b], rename error - '{e}'")
            # todo
        except Exception as e:
            rename_rejects.append(self.pngfile)
            print(
                f"[b][red]{self.pngfile}[/red][/b], parsing error - '{e}'")


def getSdMetadataTextChunk(pngfile) -> dict:
    ''' Returns the sd-metadata text chunk as a dictionary '''
    reader = png.Reader(pngfile)
    chunks = reader.chunks()

    for chunk_type, chunk_data in chunks:
        if chunk_type.decode() == 'tEXt':
            chunk_subtype, chunk_value = chunk_data.decode().split('\x00')

            if chunk_subtype == 'sd-metadata':
                return json.loads(chunk_value, strict=False)


def hasSDMetadata(pngfile: str) -> bool:
    ''' Returns true if the png file has tEXt chunk, suggesting
        that it might have SD metadata '''
    reader = png.Reader(pngfile)
    chunks = reader.chunks()

    for chunk_type, chunk_data in chunks:
        if chunk_type.decode() == 'tEXt':
            return True

    return False


def processPNG(args, pngfile):
    if hasSDMetadata(pngfile):
        mode = None

        match args.command:
            case 'chunks':
                print("-"*len(pngfile))
                print("[b][white]" + pngfile + "[/white][/b]\n")
                mode = ModeChunk(args, pngfile)
            case 'rename':
                mode = ModeRename(args, pngfile)
            case 'json':
                mode = ModeJsonDump(args, pngfile)
            case _:
                mode = ModeDefault(args, pngfile)

        mode.render()
    else:
        print(f'[red]{pngfile}[/red] - does not look like an Invoke AI image')


def main():
    parser = argparse.ArgumentParser(
        description='Swiss army knife for InvokeAI-generated PNG files')

    # accept a single-word command, required
    parser.add_argument('command', choices=['rename', 'json', 'chunks', 'sum'],
                        help='the command to run (`rename` bulk renamer, `json` dumps Invoke-specific json metadata, `chunks` dumps all text chunks, `sum` prints a one-line summary')

    # one or many files
    parser.add_argument('png_file', nargs='+',
                        help='the png files to operate on')

    # optional argument for a formatting string, defaulting to none
    parser.add_argument('--rename-format-string', '-rf',
                        default=DEFAULT_RENAME_FORMAT_STRING,
                        help=f'Format string for renaming (default: {DEFAULT_RENAME_FORMAT_STRING})')

    parser.add_argument('--dry-run', '-d',
                        action='store_true', help='Do not actually rename files, just show what might have happened.')

    parser.add_argument('--verbose', '-v',
                        action='store_true', help='Generate more verbose output')

    # add extra text to the help output

    parser.epilog = "Format string ids are pulled directly from the sub-keys of the `image` property of invoke metadata. Also available: {prompt-50}, a truncated prompt."

    # By default we just render a single-line summary, like an ls-listing

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # process the list
    for file in args.png_file:
        if not os.path.exists(file):
            print('Could not open file: ', file)

        pngfile = os.path.abspath(file)

        metadata = processPNG(args, file)

    if args.command == 'rename':
        if len(rename_rejects) > 0:
            print(
                "\n[red]----------------------------------------------------------[/red]")
            print(
                "[red]The following files were skipped and could not be renamed:[/red]")
            for file in rename_rejects:
                print(f"[red]-[/red] {file}")


if __name__ == '__main__':
    main()
