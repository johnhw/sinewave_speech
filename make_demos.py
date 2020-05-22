from sws import main

main(args=["sws.py"] + ("sounds/ex1.wav -d 4 --high 3000 --low 150 -o 4".split()))
main(args=["sws.py"] + ("sounds/ex2.wav -o 5 --window 200 --low 200".split()))
main(args=["sws.py"] + ("sounds/ex3.wav -o 4".split()))
main(args=["sws.py"] + ("sounds/ex4.wav -o 5 --high 2800 -d 8 --window 250".split()))
main(args=["sws.py"] + ("sounds/ex5.wav -d 12 --high 2000 --window 90".split()))
main(args=["sws.py"] + ("sounds/ex6.wav -d 8 --high 2500 --low 330 --window 90".split()))
main(args=["sws.py"] + ("sounds/ex7.wav --buzz 80 --window 300 -d 8 --high 2000".split()))
main(args=["sws.py"] + ("sounds/ex8.wav --noise --low 200".split()))
main(args=["sws.py"] + ("sounds/ex9.wav  --high 2800 --window 150 -o 5".split()))



