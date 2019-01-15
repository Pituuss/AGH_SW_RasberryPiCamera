# AGH_SW_RaspberryPiCamera
Project repository 

# Zaawansowane przetwarzanie obrazu w czasie rzeczywistym 

Wymagania technologiczne:

• platforma sprzętowa Raspberry Pi z dołączony modułem kamery, 

• biblioteka OpenCV. \


Repozytorium projektu: \
[https://github.com/Pituuss/AGH_SW_RaspberryPiCamera](https://github.com/Pituuss/AGH_SW_RaspberryPiCamera)


# Założenia projektu

Stworzenie programu rozpoznającego twarz, który będzie ją śledził i nakładał na nią różne efekty. Program działa na płytce Raspberry Pi i wykorzystuje kamerę openCV, która jest bezpośrednio podłączona do niego przy użyciu kabla.

Środowisko uruchomieniowe

Na początku zostało skonfigurowane odpowiednie środowisko w oparciu o instrukcję z blogu [https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/](https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/) (dostęp z dnia 14.01.2019). W wyniku podążania tą instrukcją na systemie Raspbian na płytce Raspberry Pi zainstalowane i skompilowane zostały wszelkie potrzebne zależności i biblioteki. Wśród nich znajdują się m.in. środowisko uruchomieniowe cv, python3 wraz z numpy oraz innymi przydatnymi bibliotekami. \
 \
W naszym środowisku sposób uruchamiania projektu jest następujący: \
1. Uruchomienie terminala \
2. Wpisanie komendy` `source ~/.profile`` (ustawia odpowiednie źródła)

3. Wpisanie komendy` `workon cv` `(włącza środowisko uruchomieniowe cv)

4. Przejście do projektu komendą` `cd ~/Desktop/OpenCVProject/project``

5. Uruchomienie projektu komendą ``python3 Project.py``


# Kod


## Szkic pętli głównej

Głównym zadaniem programu jest przetwarzanie obrazu, kolejne kroki wykonywane w pętli:



1.  przejęcie klatki obrazu z urządzenia (kamery)
1.  przygotowanie klatki do obróbki
    1.  usunięcie kolorów
    1.  opcjonalnie przycięcie obrazu do niższej rozdzielczości
1.  wysłanie klatki do procesu odpowiedzialnego za znalezienie twarzy
1.  odebranie wyniku działania procesu
1.  śledzenie twarzy z opcjonalnym nakładaniem efektów
1.  wyświetlanie wyników i obsługa wejścia z zewnątrz


## Omówienie poszczególnych punktów

Korzystając z biblioteki VideoStream możemy w łatwy sposób przechwytywać klatki. \



```
vs = VideoStream(src=0).start()
…
frame = vs.read()
```


Parametr `src=0 `powoduje wybranie wejścia z pierwszego napotkanego urządzenia.

Przygotowanie klatki do obróbki:  \



```
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (300, 300))
```
`Dzięki temu uzyskamy mniejszy a zatem łatwiejszy do przetworzenia o uproszczonym formacie, dla klasyfikatora nie ma znaczenia czy obrazek jest w kolorze czy nie zadziała tak samo tylko że kolory to informacje a zatem kolorowe obrazki będą przetwarzane wolniej.

Wysłanie klatki do procesu odpowiedzialnego za znalezienie twarzy. Będziemy to robić w pewnym odstępie czasu, ponieważ wykrywanie jest procesem bardzo kosztownym. \
 \
<code>sem = threading.Event() \
… \
<em>if not </em>sem.is_set() <em>and </em>time.time() - insert_time > 1.0:</code>


```
   insert_time = time.time()
   inputQueue.put(gray)
   sem.set()
```


 \
W naszym przypadku szukamy twarzy co 1 sekundę, naszym zdaniem taki czas zapewnia odpowiedni kompromis między wydajnością a płynnością rozwiązania, należy również zwrócić uwagę na zmienną wydarzenia `sem = threading.Event()` dzięki zastosowaniu mechanizmu synchronizacji unikamy tzw. 'busy waiting'. Jest to bardzo istotne z punktu widzenia wydajności programu, ponieważ wątek wykrywania twarzy nie 'szarpie się' gdy nie ma nic do zrobienia, dzięki czemu nie marnujemy taktów procesora. Kolejnym ważnym elementem jest sposób dostarczania danych do wątku `inputQueue.put(gray)` korzystamy tutaj z prostej kolejki.

Odbieranie wyniku, pary punktów oznaczających wierzchołki prostokąta, działania wątku odbywa się za pomocą \
<code> \
<em>if not </em>outputQueue.empty():</code>


```
   faces = outputQueue.get()
```


 \
Sprawdzamy tutaj czy wątek włożył już coś do kolejki, nie możemy tutaj zatrzymać pętli, ponieważ jest ona odpowiedzialna za wyświetlanie klatek.

Śledzenie twarzy jest realizowane za pomocą prostego trackera. Nie wykrywa on już twarzy on tylko podąża za wprowadzonym wycinkiem obrazu zwracając współrzędne prostokąta otaczającego dany wycinek obrazu. Korzystamy z trackera MOSSE już wbudowanego w openCV. \
 \
`tracker = cv2.TrackerMOSSE_create()`


```

if initBB is not None:
   (success, box) = tracker.update(gray)
   if success:
       (x, y, w, h) = [int(v) for v in box]
       cv2.rectangle(frame, (max(x - offset, 0), max(y - offset, 0)), (x + w + offset, y + h + offset),
                     (0, 255, 0), 2)
       x0, x1, y0, y1 = (max(y - offset, 0), y + h + offset, max(x - offset, 0), x + w + offset)
       edges = cv2.Canny(frame[x0:x1, y0:y1], 50, 50)

if faces is not None:
   if len(faces) != 0:
       initBB = tuple(faces[0])
       tracker = cv2.TrackerMOSSE_create()
       tracker.init(frame, initBB)
       faces = None
```


'initBB' jest zmienną kontrolną sprawdzającą czy tracker został już zainicjalizowany, w przypadku gdy jest wykonujemy update sprawdzając czy śledzony obiekt się przesunął, a następnie rysujemy prostokąt otaczający cel śledzenia. Natomiast tracker'a tworzymy  \
i inicjalizujemy za każdym razem gdy nasz wątek wykryje twarze, co prawda jeden tracker może śledzić tylko jeden obiekt więc my wybraliśmy że będziemy śledzić pierwszą twarz natomiast można by stworzyć więcej tracker'ów i śledzić więcej twarzy. Warto zauważyć że tracker nie operuje na uproszczonej ramce.

Wyświetlanie ramki, w tym miejscu możemy również wprowadzać efekty na obraz, czy to globalnie na cały czy też tylko na wybrany punkt. My aplikujemy filtr krawędzi na kwadracie otaczającym twarz. \
 \
<code><em>if </em>edges <em>is not </em>None:</code>


```
  frame[x0:x1,y0:y1,0] = edges
  frame[x0:x1,y0:y1,1] = edges
cv2.imshow('frame', frame)

obsługa wejścia z zewnątrz:

key = cv2.waitKey(1) & 0xff
if key == ord('q'):
   break
```


 \
Jest to prostu listener na kliknięcie klawisza. Jeśli zostanie wciśnięte 'q' zamykamy okienko i kończymy program. \
 \
<code>p.kill = <em>False</em></code>


```
sem.set()
p.join()
```


`cv2.destroyAllWindows() \
 \
`W taki sposób przerywamy nieskończoną pętlę w wątku wykrywającym twarz, czekamy aż zakończy działanie, zamykamy okienko wyświetlające obraz z kamery i kończymy program.

Opis procesu odpowiedzialnego za znalezienie twarzy:


```
p = Thread(target=classify_frame,args=(inputQueue,outputQueue,sem,kill))
p.start()
```


Pierwsza linijka tworzy nowy wątek, którego argumentami są opisane powyżej inputQueue, outputQueue, sem, kill, natomiast ciałem jest funkcja classify_frame, której opis działanie przedstawię poniżej. Następnie nowo utworzony wątek jest uruchamiany i będzie wykonywał funkcję classify_frame odpowiadającą, jak sama nazwa wskazuje, za klasyfikację ramek, do funkcji zostają przekazane argumenty z args.

Kod funkcji classify_frame wygląda następująco:


```
def classify_frame(inputQueue, outputQueue, sem, kill):
	time.sleep(2.0)
	face_cascade = 
          cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	t = threading.currentThread()
	while getattr(t, "kill", True):
		sem.wait()
		frame = inputQueue.get()
		faces = 
        face_cascade.detectMultiScale(frame, scaleFactor=1.1, 
					minNeighbors=5, minSize=(50, 50))
		outputQueue.put(faces)
		sem.clear()
```


Argumentami funkcji są inputQueue (kolejka z danymi z kamery, którą zapełnia główny wątek), outputQueue (kolejka z wykrytymi przez klasyfikator twarzami), sem (zmienna typu Event synchronizująca wątek z procesem głównym) i kill (zmienna zawierająca informację o tym, że główny proces otrzymał sygnał zakończenia.


```
time.sleep(2.0)
t = threading.currentThread()
```


Najpierw wątek jest usypiany na 2 sekundy, aby główny proces zdołał się przygotować (rozpoczął zbieranie ramek). Potem w zmiennej t zapisywany jest aktualny wątek, aby móc pobierać jego parametry.


```
while getattr(t, "kill", True):
```


Powyższy while pobiera wartość atrybutu `kill` swojego wątku i kontynuuje swoje działanie tak długo, jak jego wartość jest równa <code><em>True</em></code>. Wartość parametru <code>kill</code> zmienia główny wątek przy kończeniu działania projektu.


```
sem.wait()
frame = inputQueue.get()
```


Następnie wątek oczekuje na dane od procesu głównego i pobiera ramkę dostarczoną przez niego. Warto tutaj nadmienić, że użycie mechanizmu synchronizacji (sem to obiekt klasy Event) znacząco odciąża procesor i przyczynia się do lepszego działania programu.


```
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```


Powyższa linijka odpowiada za załadowanie gotowego klasyfikatora twarzy o nazwie `haarcascade_frontalface_default.xml`. Plik ten jest dołączony do projektu.


```
faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, 
					minNeighbors=5, minSize=(50, 50))
```


Powyższa linijka jest wywołaniem funkcji, która dla zadanej ramki zwraca listę współrzędnych prostokątów zawierających wykryte twarze albo pustą tuplę w przypadku nieznalezienia takowych (ważne jest to, aby później przy przetwarzaniu sprawdzić, czy został zwrócony choć jeden prostokąt). Oprócz ramki (frame) jako parametry przekazywane są: scaleFactor, minNeighbors oraz minSize oznaczający minimalny rozmiar ramki. Dokładniejszy opis parametrówfunkcji `detectMultiScale` dostępny jest pod linkiem: [https://docs.opencv.org/4.0.0/d1/de5/classcv_1_1CascadeClassifier.html](https://docs.opencv.org/4.0.0/d1/de5/classcv_1_1CascadeClassifier.html). 


```
outputQueue.put(faces)
sem.clear()
```


Powyższe linijki kodu odpowiadają za dodanie znalezionej twarzy do kolejki `outputQueue` oraz za poinformowanie głównego procesu o znalezionej twarzy.
