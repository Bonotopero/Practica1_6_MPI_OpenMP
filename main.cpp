#include <iostream>
#include <cstring>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include <omp.h>

using namespace std;

const char ALFABETO[36] = {
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    '0','1','2','3','4','5','6','7','8','9'
};

#define TAG_RANGOS 1
#define TAG_HALLAZGO 2
#define TAG_CANCELAR 3

struct RangoBusqueda {
    unsigned long long inicio;
    unsigned long long fin;
};

void convertirIndiceATexto(unsigned long long idx, char* buffer, int longitud) {
    for (int i = longitud - 1; i >= 0; i--) {
        buffer[i] = ALFABETO[idx % 36];
        idx /= 36;
    }
    buffer[longitud] = '\0';
}

bool validarClave(const char* clave, int longitud) {
    if (longitud == 0) return false;
    for (int i = 0; i < longitud; i++) {
        char c = clave[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) {
            return false;
        }
    }
    return true;
}

ofstream* crearLogNodo(int rank, const char* hostname) {
    char* nombreLog = new char[256];
    sprintf(nombreLog, "log_equipo_%s_nodo_%d.txt", hostname, rank);

    ofstream* logFile = new ofstream(nombreLog, ios::out | ios::trunc);
    delete[] nombreLog;

    if (logFile && logFile->is_open()) {
        (*logFile) << "==========================================================\n";
        (*logFile) << "LOG DE AUDITORIA - NODO MPI RANK: " << rank << "\n";
        (*logFile) << "NOMBRE DEL EQUIPO: Julio Angulo equipo 7\n";
        (*logFile) << "HILOS OPENMP DISPONIBLES: " << omp_get_max_threads() << "\n";
        (*logFile) << "==========================================================\n\n";
        logFile->flush();
    }
    return logFile;
}

double ejecucionSecuencial(const char* claveObjetivo, int longitud, ofstream* logFile) {
    unsigned long long totalCombinaciones = (unsigned long long)pow(36, longitud);
    char* candidato = new char[longitud + 1];

    (*logFile) << "[SECUENCIAL] Inicio de búsqueda secuencial sobre " << totalCombinaciones << " combinaciones.\n";
    logFile->flush();

    double tInicio = MPI_Wtime();
    for (unsigned long long idx = 0; idx < totalCombinaciones; idx++) {
        convertirIndiceATexto(idx, candidato, longitud);
        if (strcmp(candidato, claveObjetivo) == 0) {
            double tFin = MPI_Wtime();
            double tTotal = tFin - tInicio;
            cout << "\n[SECUENCIAL] ¡Clave encontrada!: \"" << candidato << "\" en " << tTotal << " segundos.\n";
            (*logFile) << "[SECUENCIAL] ÉXITO: Clave \"" << candidato << "\" hallada en " << tTotal << "s.\n";
            logFile->flush();
            delete[] candidato;
            return tTotal;
        }
    }

    delete[] candidato;
    return MPI_Wtime() - tInicio;
}

void ejecucionEsclavoParalelo(int rank, const char* hostname, const char* claveObjetivo, int longitud, ofstream* logFile) {
    RangoBusqueda rango;
    MPI_Recv(&rango, sizeof(RangoBusqueda), MPI_BYTE, 0, TAG_RANGOS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    (*logFile) << "[PARALELO] Rango asignado: [" << rango.inicio << " - " << rango.fin << "]\n";
    logFile->flush();

    bool encontradoLocal = false;
    bool ordenParada = false;
    MPI_Request reqCancelar;
    int flagCancelar = 0;

    MPI_Irecv(&flagCancelar, 1, MPI_INT, 0, TAG_CANCELAR, MPI_COMM_WORLD, &reqCancelar);

    double tInicio = MPI_Wtime();
    bool modoDetallado = (longitud <= 3);

    #pragma omp parallel shared(encontradoLocal, ordenParada)
    {
        int tid = omp_get_thread_num();
        char* candidatoHilo = new char[longitud + 1];

        #pragma omp for schedule(dynamic)
        for (unsigned long long idx = rango.inicio; idx <= rango.fin; idx++) {
            if (encontradoLocal || ordenParada) continue;

            if (tid == 0) {
                int testFlag = 0;
                MPI_Test(&reqCancelar, &testFlag, MPI_STATUS_IGNORE);
                if (testFlag) {
                    ordenParada = true;
                    (*logFile) << "[CANCELACIÓN] Orden de detención temprana recibida del Maestro.\n";
                    logFile->flush();
                }
            }

            convertirIndiceATexto(idx, candidatoHilo, longitud);

            if (modoDetallado && (idx % 1000 == 0)) {
                #pragma omp critical
                {
                    (*logFile) << "[Equipo: " << hostname << "] [Nodo MPI: " << rank
                               << "] [Hilo OpenMP: " << tid << "] [Candidato: " << candidatoHilo
                               << "] [Estado: Evaluando]\n";
                }
            }

            if (strcmp(candidatoHilo, claveObjetivo) == 0) {
                #pragma omp critical
                {
                    if (!encontradoLocal && !ordenParada) {
                        encontradoLocal = true;
                        double tParcial = MPI_Wtime() - tInicio;

                        (*logFile) << "[ÉXITO] [Equipo: " << hostname << "] [Nodo MPI: " << rank
                                   << "] [Hilo OpenMP: " << tid << "] [Clave Encontrada: \""
                                   << candidatoHilo << "\"] [Tiempo Parcial: " << tParcial << "s]\n";
                        logFile->flush();

                        int datosExito[2] = { rank, tid };
                        MPI_Send(datosExito, 2, MPI_INT, 0, TAG_HALLAZGO, MPI_COMM_WORLD);
                    }
                }
            }
        }
        delete[] candidatoHilo;
    }

    (*logFile) << "[FIN] Finalizó la ejecución del nodo " << rank << ".\n";
    logFile->flush();
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostLen;
    MPI_Get_processor_name(hostname, &hostLen);

    if (rank == 0) {
        cout << "==========================================================\n";
        cout << "  PRACTICA 1.6: ALGORITMO DE FUERZA BRUTA DISTRIBUIDO\n";
        cout << "  INTEGRANTES DEL EQUIPO:\n";
        cout << "  1. Angulo Díaz, Julio Abraham (Equipo 7)\n";
        cout << "==========================================================\n\n";
    }

    ofstream* logFile = crearLogNodo(rank, hostname);

    int opcion = 0;
    char* claveGuardada = new char[64];
    claveGuardada[0] = '\0';
    int longitudClave = 0;

    double tiempoSecuencial = 0.0;
    double tiempoParalelo = 0.0;

    do {
        if (rank == 0) {
            cout << "\n======================================================\n";
            cout << "  MENÚ PRINCIPAL\n";
            cout << "======================================================\n";
            cout << " Clave Actual: " << (longitudClave > 0 ? claveGuardada : "[Ninguna]") << "\n";
            cout << "------------------------------------------------------\n";
            cout << "1. Ingresar clave personalizada (A-Z, 0-9)\n";
            cout << "2. Búsqueda Secuencial (Línea Base - Nodo 0, 1 Hilo)\n";
            cout << "3. Búsqueda Paralela Distribuida (MPI + OpenMP)\n";
            cout << "4. Ver Tabla Comparativa de Tiempos y Speedup\n";
            cout << "5. Salir\n";
            cout << "Seleccione una opción: ";
            cin >> opcion;
        }

        MPI_Bcast(&opcion, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (opcion == 1) {
            if (rank == 0) {
                cout << "Ingrese la clave a descifrar (Mayúsculas y Números): ";
                cin >> claveGuardada;
                longitudClave = static_cast<int>(strlen(claveGuardada));

                if (!validarClave(claveGuardada, longitudClave)) {
                    cout << "[ERROR] La clave contiene caracteres inválidos.\n";
                    claveGuardada[0] = '\0';
                    longitudClave = 0;
                } else {
                    cout << "[OK] Clave configurada: \"" << claveGuardada << "\" (Longitud: " << longitudClave << ")\n";
                }
            }
            MPI_Bcast(&longitudClave, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (longitudClave > 0) {
                MPI_Bcast(claveGuardada, longitudClave + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
            }
        }
        else if (opcion == 2) {
            if (rank == 0) {
                if (longitudClave == 0) {
                    cout << "[AVISO] Primero debe ingresar una clave válida (Opción 1).\n";
                } else {
                    tiempoSecuencial = ejecucionSecuencial(claveGuardada, longitudClave, logFile);
                }
            }
        }
        else if (opcion == 3) {
            if (longitudClave == 0) {
                if (rank == 0) cout << "[AVISO] Primero debe ingresar una clave válida (Opción 1).\n";
                continue;
            }

            int numEsclavos = size - 1;

            if (numEsclavos == 0) {
                if (rank == 0) {
                    cout << "[INFO] Modo ejecución single-node. Asignando rango completo al Nodo 0.\n";
                    double tInicio = MPI_Wtime();
                    ejecucionSecuencial(claveGuardada, longitudClave, logFile);
                    tiempoParalelo = MPI_Wtime() - tInicio;
                }
                continue;
            }

            if (rank == 0) {
                unsigned long long totalCombinaciones = (unsigned long long)pow(36, longitudClave);
                unsigned long long chunk = totalCombinaciones / numEsclavos;
                unsigned long long rem = totalCombinaciones % numEsclavos;

                for (int i = 1; i <= numEsclavos; i++) {
                    RangoBusqueda r;
                    unsigned long long idxI = static_cast<unsigned long long>(i - 1);
                    r.inicio = idxI * chunk + min(idxI, rem);
                    r.fin = r.inicio + chunk + (idxI < rem ? 1 : 0) - 1;
                    MPI_Send(&r, sizeof(RangoBusqueda), MPI_BYTE, i, TAG_RANGOS, MPI_COMM_WORLD);
                }

                double tInicio = MPI_Wtime();
                int datosExito[2];
                MPI_Status status;

                MPI_Recv(datosExito, 2, MPI_INT, MPI_ANY_SOURCE, TAG_HALLAZGO, MPI_COMM_WORLD, &status);
                double tFin = MPI_Wtime();
                tiempoParalelo = tFin - tInicio;

                int nodoGanador = datosExito[0];
                int hiloGanador = datosExito[1];

                cout << "\n======================================================\n";
                cout << " ¡CLAVE DESCIFRADA CON ÉXITO!\n";
                cout << " Nodo Ganador (Rank): " << nodoGanador << "\n";
                cout << " Hilo OpenMP Ganador: " << hiloGanador << "\n";
                cout << " Tiempo Transcurrido: " << tiempoParalelo << " segundos.\n";
                cout << "======================================================\n";

                int banderaCancelar = 1;
                for (int i = 1; i <= numEsclavos; i++) {
                    MPI_Send(&banderaCancelar, 1, MPI_INT, i, TAG_CANCELAR, MPI_COMM_WORLD);
                }

                (*logFile) << "[PARALELO] Clave hallada por Nodo " << nodoGanador << " Hilo " << hiloGanador
                           << " en " << tiempoParalelo << "s.\n";
                logFile->flush();
            } else {
                ejecucionEsclavoParalelo(rank, hostname, claveGuardada, longitudClave, logFile);
            }
        }
        else if (opcion == 4) {
            if (rank == 0) {
                cout << "\n------------------------------------------------------\n";
                cout << "          TABLA COMPARATIVA DE RENDIMIENTO            \n";
                cout << "------------------------------------------------------\n";
                cout << " Clave Evaluada:              " << (longitudClave > 0 ? claveGuardada : "N/A") << "\n";
                cout << " Tiempo Secuencial (1 Hilo):  " << tiempoSecuencial << " s\n";
                cout << " Tiempo Paralelo Distribuido: " << tiempoParalelo << " s\n";
                if (tiempoParalelo > 0.0 && tiempoSecuencial > 0.0) {
                    cout << " Speedup Ganado (Sp):         " << (tiempoSecuencial / tiempoParalelo) << "x\n";
                } else {
                    cout << " Speedup Ganado (Sp):         N/A (Requiere ejecutar Opción 2 y 3)\n";
                }
                cout << "------------------------------------------------------\n";
            }
        }

    } while (opcion != 5);

    delete[] claveGuardada;

    if (logFile) {
        (*logFile) << "Cierre formal de procesos MPI.\n";
        logFile->close();
        delete logFile;
    }

    if (rank == 0) {
        cout << "\n==========================================================\n";
        cout << "  FIN DE LA EJECUCIÓN DEL PROGRAMA\n";
        cout << "  INTEGRANTES DEL EQUIPO:\n";
        cout << "  1. Angulo Díaz, Julio Abraham (Equipo 7)\n";
        cout << "==========================================================\n";
    }

    MPI_Finalize();
    return 0;
}
