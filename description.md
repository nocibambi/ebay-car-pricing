A házi feladat megoldásához két adathalmazt osztunk meg veletek. A tanuló adathalmazon ismertek a célváltozó értékei, míg a teszthalmazon nem. A modelledet - figyelve a túltanulás elkerülésére - a tanuló halmazon kell létrehozni, majd ezt a modellt alkalmazni, végül az így kapott megoldást beküldeni a Kaggle platformon.
Rendelkezésre álló adathalmazok

* autos_training_final.csv - a tanuló adathalmaz
* autos_testing_final.csv - a modell alkalmazásához használható halmaz
* autos_submission.csv - a beküldésnél ebben a formátumban kell a megoldást előállítani

Változók jelentése

Az adathalmaz az Ebay Kleinanzeigen oldalról származik, köszönet érte a Kaggle platformnak.

* dateCrawled: a mentés dátuma, minden további változó ezen a dátumon alapul
* name : az autó neve
* seller : private (magán) vagy dealer (kereskedő)
* offerType
* vehicleType: az autó típusa
* yearOfRegistration: melyik évben regisztrálták az autót
* gearbox
* powerPS : lóerő
* model: modell
* kilometer: kilométerek száma
* monthOfRegistration: melyik hónapban regisztrálták az autót
* fuelType: üzemanyag típusa
* brand: márka
* notRepairedDamage: van-e nem javított kár az autón
* dateCreated: mikor jött létre a hirdetés az ebay oldalán
* nrOfPictures: a hirdetéshez tartozó képek száma
* postalCode
* label: célváltozó, az autó ára többet ér-e, mint a kategória átlagára
* lastSeenOnline: utolsó előfordulás
