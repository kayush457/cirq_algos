import random
import cirq
def make_quantum_teleportation_circuit(ranX,ranY):
    circuit=cirq.Circuit()
    msg,alice,bob=cirq.LineQubit.range(3)
    circuit.append([cirq.H(alice),cirq.CNOT(alice,bob)])
    circuit.append([cirq.X(msg)**ranX,cirq.Y(msg)**ranY])
    circuit.append([cirq.CNOT(msg,alice),cirq.H(msg)])
    circuit.append(cirq.measure(msg,alice))
    circuit.append([cirq.CNOT(alice,bob),cirq.CZ(msg,bob)])
    return msg,circuit
def main():
    ranX=random.random()
    ranY=random.random()
    msg,circuit=make_quantum_teleportation_circuit(ranX,ranY)
    sim=cirq.Simulator()
    message=sim.simulate(cirq.Circuit([cirq.X(msg)**ranX,cirq.Y(msg)**ranY]))
    print("Bloch sphere of Alice's qubit: ")
    b0X,b0Y,b0Z=cirq.bloch_vector_from_state_vector(message.final_state,0)
    print("x: ",round(b0X,4),"y: ",round(b0Y,4),"z: ",round(b0Z,4))
    print("\nCircuit")
    print(circuit)
    final_results=sim.simulate(circuit)
    print("\nBloch sphere of Bob's qubit: ")
    b2X,b2Y,b2Z=cirq.bloch_vector_from_state_vector(final_results.final_state,2)
    print("x: ",round(b2X,4),"y: ",round(b2Y,4),"z: ",round(b2Z,4))
if __name__=="__main__":
    main()
