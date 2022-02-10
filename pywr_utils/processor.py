"""
    Module for creating polyvis-compatible input files
"""
import os
import sys
import pandas as pd
import numpy as np
import json
from collections import defaultdict

import logging
logging.basicConfig(level=logging.DEBUG)

logging.getLogger('parso').setLevel(logging.WARNING)

class Predicates:
    LT = 0
    GT = 1
    EQ = 2
    LE = 3
    GE = 4

_predicate_lookup = {
    "LT": Predicates.LT, "<": Predicates.LT,
    "GT": Predicates.GT, ">": Predicates.GT,
    "EQ": Predicates.EQ, "=": Predicates.EQ,
    "LE": Predicates.LE, "<=": Predicates.LE,
    "GE": Predicates.GE, ">=": Predicates.GE,
}

class PywrModelProcessor:
    """
        Consume a pywr model and perform operations such as removing or adding nodes, parameters or recorders

    """
    def __init__(self, model_file):

        self.model_file = model_file
        self.model_path = model_file
        if isinstance(model_file, str):
            with open(self.model_file, 'r') as f:
                self.model = json.load(f)
        else:
            self.model = json.load(self.model_file)
            self.model_path = self.model_file.name

        self.file_location = os.path.dirname(os.path.expanduser(__file__))

        self.all_parameter_names = list(self.model['parameters'].keys())
        #create a parameters dict with normalised keys for more reliable
        #matching of parameters
        self.model_parameters = dict(zip(self.normalise_name(self.all_parameter_names), self.model['parameters'].values()))
        self.model_parameter_name_lookup = dict(zip(self.normalise_name(self.all_parameter_names), self.all_parameter_names))
        self.output_dir = os.path.split(self.model_path)[0]

        self.log = logging.getLogger('pywr-utils')
        self.log.setLevel('DEBUG')

        self.decision_variables = []
        self.aggregate_multipliers = {}
        self.rectifier_parameters = []
        self.constant_parameters = []
        self.dec_node_map = defaultdict(list) # map from decision variable to node(s)
        self.model_constants = {}

        # a map from parameter names which are directly set on nodes to avoid
        #iterating over ndes looking for pggarameters constantly
        self.param_node_reverse_lookup = defaultdict(list)

        self.agg_params = {}
        self.dependency_params = {}
        #a mapping from a parameter's normalised name to its original name from the model
        self.param_normalised_name_map = dict(zip(self.model_parameters.keys(), self.model['parameters'].keys()))

        #for the special indexearrayparameters this is a mapping from the index variable
        #to all paramters which use it as the index parameter
        self.index_param_lookup = defaultdict(list)

        self.parameter_node_map = defaultdict(list)
        self.node_decision_var_map = defaultdict(list)
        #a lookup from decision variable to the aggregated parameter to which it belongs
        self.dec_variable_aggregated_lookup = defaultdict(list)

        self.rectifier_df = None
        self.indexparam_df = None

        self.plan_selected_options = defaultdict(list)

        self.matched_nodes = defaultdict(list)

        tmp_name_map = {}
        #now normalises the keys so they're easier to look up@
        self.name_map = dict([(self.normalise_name(k), tmp_name_map[k]) for k in tmp_name_map])

    def get_name(self, name):
        """
            Get the name to put into the file. Sometimes the name going into the file
            is different to that in the model, so it must go through this funciton
        """
        #leaving this in, just in case we need to manipulate names in the future
        return name
        #return self.name_map.get(self.normalise_name(name), name)

    def normalise_name(self, name):
        """
            make a string lower-case and remove _ - and spaces
        """
        try:
            float(name)
            return name
        except:
            pass

        if not isinstance(name, str): # it must be a list
            return [self.normalise_name(n) for n in name]
        else:
            return name.lower().replace('_', '').replace(' ', '').replace('-', '').replace('.', '')

    def get_all_agg_params(self):
        """
            Get all aggregate parameters
        """
        self.log.info("Getting all aggregate parameters")
        for p_name, p in self.model_parameters.items():
            if p['type'].find('aggregate') == 0:
                self.agg_params[p_name] = p
            elif p['type'] in ('indexedarrayparameter', 'indexedarray'):
                normalised_name = self.normalise_name(p['index_parameter'])
                self.index_param_lookup[normalised_name].append(p_name)
            elif p['type'] in ('controlcurveindex', 'controlcurveindexparameter'):
                for c in p['control_curves']:
                    #we're only interested in parameter references
                    if isinstance(c, str):
                        normalised_name = self.normalise_name(c)
                        self.index_param_lookup[normalised_name].append(p_name)
            elif p['type'] == 'DependencyConstraint':
                self.dependency_params[p_name] = p

    def make_dec_variable_aggregated_lookup(self):
        """
            Make a lookup from all rectifier and constant parameters to the
            aggregated functions they are associated to
        """

        for paramname, param in self.model['parameters'].items():
            if param['type'] == 'aggregated':
                for sp in param['parameters']:
                    try:
                        float(sp)
                        continue
                    except:
                        pass
                    if not isinstance(sp, str):
                        continue
                    normalised_name = self.normalise_name(sp)
                    self.dec_variable_aggregated_lookup[normalised_name].append(paramname)

    def make_parameter_node_lookup(self):
        for n in self.model['nodes']:
            self.get_node_parameter_lookup(n)


    def get_node_parameter_lookup(self, node):

        for k, v in node.items():
            if v in self.all_parameter_names:
                self.parameter_node_map[self.normalise_name(v)].append(node['name'])
                if self.model['parameters'][v].get('is_variable') == True:
                    self.node_decision_var_map[self.get_name(node['name'])].append(self.normalise_name(v))

                self.get_sub_parameters(v, node['name'])

    def get_sub_parameters(self, parameter_name, node_name):
        """
            Get the name of all the sub-parameters of this parameter and associate them
            to the specified node
        """

        param = self.model['parameters'][parameter_name]

        if param.get('is_variable') is True:
            self.node_decision_var_map[self.get_name(node_name)].append(self.normalise_name(parameter_name))


        paramkeys = ['parameter', 'capacity_param', 'max_flow', 'index_parameter', 'control_curve']
        for paramkey in paramkeys:
            if param.get(paramkey) and isinstance(param[paramkey], str):
                self.parameter_node_map[self.normalise_name(param[paramkey])].append(node_name)
                self.get_sub_parameters(param[paramkey], node_name)

        for key in ('params', 'parameters', 'control_curves'):
            for sub_param in param.get(key, []):
                if isinstance(sub_param, str):
                    self.parameter_node_map[self.normalise_name(sub_param)].append(node_name)
                    self.get_sub_parameters(sub_param, node_name)

        for k, v in param.items():
            if k.find('parameter') >= 0 and isinstance(v, str) and v in self.model['parameters']:
                self.parameter_node_map[self.normalise_name(v)].append(node_name)
                self.get_sub_parameters(v, node_name)

    def get_parameter_node(self, parameter_name, original_parameter=None):
        """
            For a given parameter, find the node to which it relates by looking
            up the parameter tree until it finds a node.
        """

        normalised_name = self.normalise_name(parameter_name)
        if self.model_parameters.get(normalised_name) is None:
            self.log.warning("Parameter %s not found in the model.", parameter_name)
            return []

        #is the parameter defined directly on a node?
        if normalised_name in self.param_node_reverse_lookup:
            param_node_names = self.param_node_reverse_lookup[normalised_name]
            return param_node_names

        if self.model_parameters[normalised_name]['type'] in ('indexvariable', 'aggregatedindex', 'controlcurveindex', 'controlcurveindexparameter'):
            for indexed_param in self.index_param_lookup[normalised_name]:
                #pass the original decision variable name up the tree
                return self.get_parameter_node(indexed_param)

        for dep_param_name, dep_param in self.dependency_params.items():
            if normalised_name == self.normalise_name(dep_param['capacity_param']):
                #pass the original decision variable name up the tree
                return self.get_parameter_node(self.normalise_name(dep_param_name))


        # if normalised_name in self.agg_params:
        #     return [self.get_parameter_node(i) for i in self.index_param_lookup[normalised_name]]
        # for agg_param_name, agg_params in self.dec_variable_aggregated_lookup.items():
        #     import pudb; pudb.set_trace()
        #     if normalised_name in [self.normalise_name(p) for p in agg_params]:
        #         #pass the original decision variable name up the tree
        #         return self.get_parameter_node(self.normalise_name(agg_param_name))

        for agg_param_name, agg_param in self.agg_params.items():
            if normalised_name in [self.normalise_name(p) for p in agg_param['parameters']]:
                #pass the original decision variable name up the tree
                return self.get_parameter_node(self.normalise_name(agg_param_name))

        #final check...look at all the parameters and see if this parameter is referenced by a 'max_flow', as
        #in the case of trent_rutland_capacity_decision
        for param_name, param in self.model_parameters.items():
            if param.get('max_flow'):
                if self.normalise_name(param['max_flow']) == normalised_name:
                    return self.get_parameter_node(self.normalise_name(param_name))

    def get_aggregated_multipliers(self, parameter_name):
        """
            Trace through the parameters for all the multipliers associated to an
            aggregated node.
        """

        agg_param = self.model_parameters[parameter_name]

        multipliers = []

        for sub_parameter in agg_param['parameters']:
            sub_parameter = self.normalise_name(sub_parameter)
            try:
                float(sub_parameter)
                val = multipliers.append(sub_parameter)
            except:
                val = self.get_parameter_value(sub_parameter)
            multipliers.append(val)
        return multipliers

    def get_parameter_value(self, parameter_name):
        """
            get the value of a parameter
        """

        #is an int or float being passed in?
        try:
            float(parameter_name)
            return parameter_name
        except:
            pass

        param = self.model_parameters[self.normalise_name(parameter_name)]
        val = param.get('value')
        p_type = param['type']
        if p_type == 'constant':
            if parameter_name in self.constant_df.columns:
                return self.constant_df[parameter_name]
            else:
                return self.model_constants[parameter_name]
        elif p_type == 'rectifier':
            return self.rectifier_df[parameter_name]
        elif p_type in ('aggregated', 'aggregatedparameter'):
            return self.do_aggregation(param['agg_func'], param['parameters'])
        elif p_type == 'DependencyConstraint':
            return self.get_parameter_value(self.normalise_name(param['capacity_param']))
        elif p_type == 'NorthMuskhamAbstraction':
            return self.get_parameter_value(self.normalise_name(param['max_flow']))
        elif p_type == 'Middle_Level_at_St_Germans_Fenland_abs':
            return 99999
        elif p_type in ('indexedarrayparameter', 'indexedarray'):
            if parameter_name in self.indexparam_df.columns:
                return self.indexparam_df[parameter_name]
            else:
                return 1
            #return [self.get_parameter_value(self.normalise_name(p)) for p in param['params']]
        elif p_type.lower() in (
                'monthlyprofile',
                'dailyprofile',
                'azuredatalakedataframe',
                'controlcurve',
                'middle_level_at_st_germans_fenland_abs',
                'bedford_ouse_at_earith_to_fenland',
                'slrwithambostonabstraction',
                'sffd_slr_rule_current_ts'
                ):
            return 1
        elif p_type == 'negative':
            return self.get_parameter_value(self.normalise_name(param['parameter'])) * -1
        elif p_type == 'min':
            return min(self.get_parameter_value(param['parameter']), param['threshold'])
        elif p_type == 'max':
            return max(self.get_parameter_value(param['parameter']), param['threshold'])
        elif p_type.lower() == 'parameterthreshold':
            threshold = param['threshold']
            predicate = _predicate_lookup[param['predicate']]
            paramval = self.get_parameter_value(self.normalise_name(param['parameter']))
            meets_threshold = False
            if predicate == Predicates.LT:
                meets_threshold = paramval < threshold
            elif predicate == Predicates.GT:
                meets_threshold = paramval > threshold
            elif predicate == Predicates.LE:
                meets_threshold = paramval <= threshold
            elif predicate == Predicates.GE:
                meets_threshold = paramval >= threshold
            else:
                meets_threshold = x == threshold

            if meets_threshold is True:
                if param.get('values') is None:
                    return 1
                else:
                    return param['values'][1]
            else:
                if param.get('values') is None:
                    return 0
                else:
                    return param['values'][0]


        self.log.critical("Can't get value of parameter %s, Param type %s not recognised",
                          parameter_name, param['type'])
        exit(1)

    def get_param_node_reverse_lookup(self):
        """
            Get all parameters defined directly on nodes and make a mapping
            from their name to the node(s) on which they are defined.
        """
        self.log.info("Making parameter-node reverse-lookup")
        for n in self.model['nodes']:
            for val in n.values():
                if not isinstance(val, str):
                    continue

                norm_val = self.normalise_name(val)

                if norm_val in self.model_parameters.keys():
                    self.param_node_reverse_lookup[norm_val].append(n['name'])

    def aggregate_parameters(self, param_name):
        """
            Perform an aggregation function on a list of parameters
        """
        param_name = self.normalise_name(param_name)
        param = self.model_parameters[param_name]
        agg_func = param['agg_func']
        total = self.do_aggregation(agg_func, param['parameters'])
        return total

    def do_aggregation(self, agg_func, agg_list):
        """
        """
        total = None
        for item in agg_list:
            try:
                val = float(item)
            except:
                n = self.normalise_name(item)
                if n in self.rectifier_df.columns:
                    val = self.rectifier_df[self.normalise_name(item)]
                else:
                    val = self.get_parameter_value(n)

            if total is None:
                total = val
            else:
                if agg_func == 'product':
                    total = total * val
                elif agg_func == 'sum':
                    total = total + val
                elif agg_func == 'min':

                    if isinstance(val, pd.Series) and not isinstance(total, pd.Series):
                        x = pd.Series(data=[total]*len(val), index=val.index)
                        total = pd.concat([x, val], axis=1).min(axis=1)
                    elif isinstance(total, pd.Series) and not isinstance(val, pd.Series):
                        x = pd.Series(data=[val]*len(total), index=total.index)
                        val = pd.concat([x, total], axis=1).min(axis=1)
                    else:
                        if val < total:
                            total = val
                elif agg_func == 'max':
                    if isinstance(val, pd.Series) and not isinstance(total, pd.Series):
                        x = pd.Series(data=[total]*len(val), index=val.index)
                        total = pd.concat([x, val], axis=1).max(axis=1)
                    elif isinstance(val, pd.Series) and isinstance(total, pd.Series):
                        total = pd.concat([total, val], axis=1).max(axis=1)
                    elif isinstance(total, pd.Series) and not isinstance(val, pd.Series):
                        x = pd.Series(data=[val]*len(total), index=total.index)
                        val = pd.concat([x, total], axis=1).min(axis=1)
                    else:
                        if val < total:
                            total = val
                else:
                    raise Exception(f"Unknown function {agg_func}")

        return total

    def all_numerical(self, paramlist):
        """
        Check if all the elements in a list are numerical
	"""
        try:
            [float(p) for p in paramlist]
            return True
        except:
            return False

    def get_max_flow(self, node):
        """
            Now that all the data has been compiled, calculate the max flow of a node
            using the 'get_parameter_value' function
        """

        if node.get('max_volume'):
            max_cap_param = self.normalise_name(node['max_volume'])
        else:
            max_cap_param = self.normalise_name(node['max_flow'])

        max_cap_value = self.get_parameter_value(max_cap_param)

        return max_cap_value

    def identify_node_type(self, nodename):
        """
        Try to identify the type of a node by its name.
        The code here is the option code used by Hydra, and is what will become
        the metric code
        """
        name = nodename.lower()
        if name.endswith('sw'):
            return 'SW'
        if name.endswith('gw'):
            return 'GW'
        if name.endswith('abstraction') or name.find('abstraction') >= 0:
            return 'ABSTRACTION'
        if name.endswith('reservoir'):
            return 'RESER'
        if name.endswith('reuse') or name.find('efr') >= 0:
            return 'REUSE'
        if name.find('wtw') >= 0:
            return 'WTW'
        if name.endswith('dtt'):
            return 'DTT'
        if name.find('desalination') >= 0:
            return 'DESAL'
        if name.find('bcttw') >= 0:
            return 'TRNS'
        if name.endswith('transfer'):
            return 'TRNS'
        if name.find('_to_') >= 0:
            return 'TRNS'
        if name.find('2a-') >= 0:
            return 'TRNS'
        if name.find('tankering') >= 0:
            return 'TANKERING'
        if name.find('expansion') >= 0:
            return 'EXPANSION'
        if name.find('affinity export') >= 0:
            return 'AFEXP'
        if ' to ' in name:
            return 'TRNS'

        self.log.critical("Could not recognise the type of node %s", nodename)
        sys.exit(1)

    def make_model_constants_map(self):
        for pname, p in self.model_parameters.items():
            if p['type'] == 'constant':
                if p.get('value') is None:
                    return 1
                self.model_constants[pname] = p['value']

    def process_model(self):
        """
            Go through each of the decision variables, match them up to a
            parameter in the model, and then match those up to options in the
            model. Then decide, for each plan, if that option was chosen or not.
        """
        #map all parameters to nodes, even those not defined directly on a node
        self.make_parameter_node_lookup()

        self.get_param_node_reverse_lookup()
        self.get_all_agg_params()
        self.make_dec_variable_aggregated_lookup()
        self.make_model_constants_map()
        for d in self.decision_variables:
            self.get_parameter_node(d)

        if self.metrics_only is not True:
            self.get_node_max_capacities()
            self.aggregate_node_types()

        self.remove_decision_columns()


    def save(self, output_dir=None):
        """
        Save pywr json data to the specified directory
        """
        data_dir = output_dir
        if output_dir is None:
            self.log.warning("No data dir specified. Returning.")

        title = self.model['metadata']['title'].replace(' ', '_')
        if output_dir is None:
            output_dir = self.output_dir
        #check if the output folder exists and create it if not
        if not os.path.isdir(output_dir):
            #exist_ok sets unix the '-p' functionality to create the whole path
            os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f'modified_{title}.json')
        with open(filename, mode='w') as fh:
            json.dump(self.model, fh, sort_keys=True, indent=2)

        self.log.info(f'Successfully exported "{filename}"')
        return filename

    def remove_orphan_parameters(self):
        """
            Remove any parameters from the model which are not referred to by any
            node or by any other parameter.
            This iterates through each parameter and checks where it has been referenced
            from. If it can't find a reference, it is removed.
        """
        self.log.info("Removing orphan parameters.")
        # self.get_param_node_reverse_lookup()
        # self.get_all_agg_params()
        # self.make_dec_variable_aggregated_lookup()
        self.make_parameter_node_lookup()

        node_names = [n['name'] for n in self.model['nodes']]

        params_to_remove = []
        for p in self.model['parameters']:
            if self.normalise_name(p) not in self.parameter_node_map:
                params_to_remove.append(p)
        for param_to_remove in params_to_remove:
            self.log.info(f"removing parameter {param_to_remove}")
            del(self.model['parameters'][param_to_remove])
        recorders_to_remove = []
        for rname, r in self.model['recorders'].items():
            if r.get('node') is not None:
                if r['node'] not in node_names:
                    recorders_to_remove.append(rname)
        for recorder_to_remove in recorders_to_remove:
            self.log.info(f"removing recorder {recorder_to_remove}")
            del(self.model['recorders'][recorder_to_remove])

    def check_node_references(self):
        """
            Check the parameters and recorders of a model to see if all the nodes
            they refer to are in the model
        """
        node_names = [n['name'] for n in self.model['nodes']]

        for pname, parameter in self.model['parameters'].items():
            for k, v in parameter.items():
                if k.find('node') >= 0  and k.find('parameter') == -1:
                    if v not in node_names:
                        self.log.warning(f"Unreferenced node in parameter {pname} {k} : {v}")
        for rname, recorder in self.model['recorders'].items():
            for k, v in recorder.items():
                if k.find('node') >= 0:
                    if v not in node_names:
                        self.log.warning(f"Unreferenced node in recorder {pname} {k} : {v}")
