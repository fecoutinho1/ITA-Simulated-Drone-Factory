import seaborn as sns
import matplotlib.pyplot as plt
import simpy
import random
import pandas as pd
from datetime import datetime, timedelta
import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
import igraph as ig  
from collections import defaultdict
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

## scenarios
cenarios_parametros = {
    "Cenário 1": {"manutencao": 2.0, "pintura": 1.5, "montagem": 1.1},
    "Cenário 2": {"manutencao": 1.0, "pintura": 1.5, "montagem": 1.1},
    "Cenário 3": {"manutencao": 2.0, "pintura": 1.3, "montagem": 1.1},
    "Cenário 4": {"manutencao": 2.0, "pintura": 1.5, "montagem": 1.0},
    "Cenário 5": {"manutencao": 1.0, "pintura": 1.3, "montagem": 1.0}
}

for nome_cenario, tempos in cenarios_parametros.items():
    print(f"\n{'='*50}")
    print(f"🔹 {nome_cenario} 🔹")
    print(f"{'='*50}")

    event_log = []

    def log_event(product_id, event, time_stamp_ini, time_stamp_fim, resource):
        event_log.append({
            'Product_id': product_id,
            'Machine_id': event,
            'Time_stamp_Ini': time_stamp_ini,
            'Time_stamp_Fim': time_stamp_fim,
            'Simulated_time': time_stamp_fim,
            'Resource': resource
        })

    class Drone_Factory:
        def __init__(self, env):
            self.plastic = simpy.Container(env, capacity=1000, init=500)
            self.electronic = simpy.Container(env, capacity=100, init=100)
            self.first_assembler_body = simpy.Container(env, capacity=100, init=0)
            self.first_assembler_helice = simpy.Container(env, capacity=100, init=0)
            self.second_assembler_body = simpy.Container(env, capacity=200, init=0)
            self.second_assembler_helice = simpy.Container(env, capacity=200, init=0)
            self.dispatch = simpy.Container(env, capacity=500, init=0)

    def body_maker(env, drone_factory):
        case_id = 1
        while True:
            time_stamp_ini = env.now
            log_event(case_id, 'Start Body Making', time_stamp_ini, None, 'body_maker')
            yield drone_factory.plastic.get(1)
            yield env.timeout(1)
            yield drone_factory.first_assembler_body.put(1)
            time_stamp_fim = env.now
            log_event(case_id, 'End Body Making', time_stamp_ini, time_stamp_fim, 'body_maker')
            case_id += 1

    def helice_maker(env, drone_factory):
        case_id = 1
        while True:
            time_stamp_ini = env.now
            log_event(case_id, 'Start Helice Making', time_stamp_ini, None, 'helice_maker')
            yield drone_factory.plastic.get(1)
            yield env.timeout(1)
            yield drone_factory.first_assembler_helice.put(4)
            time_stamp_fim = env.now
            log_event(case_id, 'End Helice Making', time_stamp_ini, time_stamp_fim, 'helice_maker')
            case_id += 1

    def painter(env, drone_factory):
        case_id = 1
        while True:
            time_stamp_ini = env.now
            log_event(case_id, 'Start Painting', time_stamp_ini, None, 'painter')
            yield drone_factory.first_assembler_body.get(2)
            yield drone_factory.first_assembler_helice.get(8)
            yield env.timeout(tempos["pintura"])
            yield drone_factory.second_assembler_helice.put(8)
            yield drone_factory.second_assembler_body.put(2)
            time_stamp_fim = env.now
            log_event(case_id, 'End Painting', time_stamp_ini, time_stamp_fim, 'painter')
            case_id += 1

    def assembler(env, drone_factory):
        case_id = 1
        while True:
            time_stamp_ini = env.now
            log_event(case_id, 'Start Assembling', time_stamp_ini, None, 'assembler')
            yield drone_factory.second_assembler_helice.get(4)
            yield drone_factory.second_assembler_body.get(1)
            yield drone_factory.electronic.get(1)
            yield env.timeout(tempos["montagem"])
            yield drone_factory.dispatch.put(1)
            time_stamp_fim = env.now
            log_event(case_id, 'End Assembling', time_stamp_ini, time_stamp_fim, 'assembler')
            case_id += 1

    env = simpy.Environment()
    drone_factory = Drone_Factory(env)
    env.process(body_maker(env, drone_factory))
    env.process(helice_maker(env, drone_factory))
    env.process(painter(env, drone_factory))
    env.process(assembler(env, drone_factory))
    env.run(until=40)  # 5 dias * 8 horas

    event_log_df = pd.DataFrame(event_log)
    print("\nEvent Log Summary:")
    print(event_log_df.head())
    print(f"\nTotal events: {len(event_log_df)}")

    log_pm4py = event_log_df.copy()

    log_pm4py = log_pm4py.rename(columns={
        'Product_id': 'case:concept:name',
        'Machine_id': 'concept:name',
        'Simulated_time': 'time:timestamp'
    })

    log_pm4py['time:timestamp'] = pd.to_datetime(log_pm4py['time:timestamp'], unit='h', origin=pd.Timestamp('2024-01-01'))

    from pm4py.objects.log.util import dataframe_utils
    from pm4py.objects.conversion.log import converter as log_converter

    log_pm4py = dataframe_utils.convert_timestamp_columns_in_df(log_pm4py)
    event_log = log_converter.apply(log_pm4py)

    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

    heu_net = heuristics_miner.apply_heu(event_log)

    gviz = hn_visualizer.apply(heu_net)
    hn_visualizer.save(gviz, f"heuristics_net_{nome_cenario}.png")

    from pm4py.objects.conversion.heuristics_net import converter as hn_converter
    petri_net, initial_marking, final_marking = hn_converter.apply(heu_net)

    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    replayed_traces = token_replay.apply(event_log, petri_net, initial_marking, final_marking)

    from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
    from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
    from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator

    fitness = replay_fitness.evaluate(replayed_traces)
    print(f"\nProcess Mining Metrics:")
    print(f"Fitness: {fitness}")

    precision = precision_evaluator.apply(event_log, petri_net, initial_marking, final_marking)
    print(f"Precision: {precision}")

    generalization = generalization_evaluator.apply(event_log, petri_net, initial_marking, final_marking)
    print(f"Generalization: {generalization}")

    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    gviz = pn_visualizer.apply(petri_net, initial_marking, final_marking, parameters={"format": "png"})
    pn_visualizer.save(gviz, f"petri_net_{nome_cenario}.png")
    
    print(f"\nVisualizations saved as:")
    print(f"- heuristics_net_{nome_cenario}.png")
    print(f"- petri_net_{nome_cenario}.png")