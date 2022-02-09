import click
import json
import os
import pandas
from .processor import PywrModelProcessor
from pywr.model import Model

def start_cli():
    cli()


@click.group()
def cli():
    pass

@cli.command()
@click.argument('filename', default=None)
def run_file(filename):
    """ Run pywr on the specified file """

    model = Model.load(filename)
    model.setup()
    model.run()
    click.echo(f'Pywr model run success!')

@cli.command()
@click.argument('model', type=click.File())
@click.option("-o", "--output-dir", type=click.File(), default=None)
def remove_orphan_parameters(model, output_dir):
    """Remove any parameters from a model which are not referred to by any
    other entity in the model"""
    processor = PywrModelProcessor(model)

    processor.remove_orphan_parameters()

    processor.save(output_dir=output_dir)
