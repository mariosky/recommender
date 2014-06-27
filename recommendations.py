##
##  
##  Implemetacion de Algoritmos de recomendacion
##  Basados en el Libro de PCI de Toby Segaran.
##       

import numpy
from math import sqrt
import sqlite3
import datetime
from MovielensDB import *

#from FIS import Recommender
#from FIS import RecommenderItemCF

import matplotlib.pyplot as plt
#import RecommenderEval
#import RecommenderItemCF2



#####
#####  FUNCIONES DE SIMILARIDAD
#####



def sim_distance(prefs,person1,person2):
  # Get the list of shared_items
  if person1 not in prefs or person2 not in prefs:  return 0 , 0

  si={}

  for item in prefs[person1]:
    if item in prefs[person2]:
        si[item]=1

  items_in_common = len(si)
  # if they have no ratings in common, return 0
  if items_in_common == 0 : return 0 , 0

  # Add up the squares of all the differences
  sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2)
                      for item in prefs[person1] if item in prefs[person2]])
  return 1/(1+sum_of_squares), items_in_common

def sim_pearson2(prefs,p1,p2):
  # Get the list of mutually rated items
  if p1 not in prefs or p2 not in prefs:  return 0 , 0
  si={}
  for item in prefs[p1]:
    if item in prefs[p2]: si[item]=1

  # if they are no ratings in common, return 0
  items_in_common = len(si)
  if items_in_common == 0 : return 0 , 0

  # Sum calculations
  n=items_in_common

  # Sums of all the preferences
  sum1=sum([prefs[p1][it] for it in si])
  sum2=sum([prefs[p2][it] for it in si])

  # Sums of the squares
  sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
  sum2Sq=sum([pow(prefs[p2][it],2) for it in si])

  # Sum of the products
  pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])

  # Calculate r (Pearson score)
  num=pSum-(sum1*sum2/n)
  den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
  if den==0: return 0 , 0
  r=num/den
  return r , items_in_common
#prefs={1:{1:1,2:2,3:3,4:5,5:8},2:{1:0.11,2:0.12,3:0.13,4:0.15,5:0.18}}

def sim_pearson(prefs,p1,p2):
  # Get the list of mutually rated items
  if p1 not in prefs or p2 not in prefs:  return 0 , 0
  si={}
  for item in prefs[p1]:
    if item in prefs[p2]: si[item]=1

  # if they are no ratings in common, return 0
  items_in_common = len(si)
  if items_in_common == 0 : return 0 , 0

  # Sum calculations
  n=items_in_common
  p1_avg = user_average(prefs,p1)
  p2_avg = user_average(prefs,p2)

  num=sum( [ (prefs[p1][it]-p1_avg) *(prefs[p2][it]-p2_avg) for it in si])
  sum1=sum( [ pow( prefs[p1][it]-p1_avg ,2) for it in si])
  sum2=sum( [ pow( prefs[p2][it]-p2_avg ,2) for it in si])
  # Sums of all the preferences
  
 # Calculate r (Pearson score)

  den=sqrt(sum1)*sqrt(sum2)
  if den==0: return 0 , 0
  r=num/den
  return r , items_in_common

#############################################################################################
# cargar la base de datos de similitudes (similitud de coseno)
# conectar la base de datos
# item1 = primer nunmero de pelicula
# item2 = segundo numero de pelicula
# similitud = similutud que ahi entre la primer pelicula(item1) y todas las demas peliculas (item2)

def cargarBDDTFIDF():
 conexion = sqlite3.connect('TFIDF.sqlite')
 cur = conexion.cursor()
 # consulta sql
 cur.execute('SELECT indice, item1, item2, similitud FROM SimTFIDF')
 dic={}
 # Muestra todos los datos de la consulta
 for (indice, item1, item2, similitud) in cur:
   dic.setdefault(int(item1),{})
   dic[int(item1)][int(item2)]=float (similitud)
 return dic
##############################################################################################
#algoritmo del filtrado basado en contenido
#prefs= es la tabla de ratings en forma de diccionario: contiene el usuario con la speliculas que ha votado
#user = es el usuario al cual se le quiere predecir un item
#item = es el item que se quiere predecir

def basadocontenido(prefs, user, item):
  #si el usuario no tiene item votados termina el algortimo y regresa un None
  if len (prefs[user])== 0:
    return None
  prediccion=0
  contador=0
  #carga la base de datos con las similitudes TFIDF
  similitud=cargarBDDTFIDF()
  #por cada item votado por el usuario
  for otro in prefs[user]:
    #se suman la similitu por el voto de cada item votado por el usuario
    prediccion=prediccion + (similitud[item][otro]*prefs[user][otro])
    contador=contador+1
    #regresa el promedio de las similitudes por los votos
  return prediccion/contador


#############################################################################
#                   ## Martes 31 de enero de 2011##                         #
#aplicar la similitud de coseno a la tabla tfidf
def aplicar_similitud():
  #aplicar la similitud y pasar como parametro el metodo que queramos
  # como estandarizar la funcion
  conn = sqlite3.connect('TFIDF.sqlite')
  c = conn.cursor()
  conn.text_factory = str
  c.execute(
    """
    CREATE TABLE SimTFIDF
    (
     indice integer primary key,
     item1 text,
     item2 text,
     similitud real
    )
    """)
  # en la base de datos TFIDF.squite se guardo la tabla de similitudes que lleva por nombre SimTFIDF
  #cargar La tabla TFIDF con la funcion cargar tfidf
  print 'Cargando Tabla TFIDF a Diccionario'
  TFIDF = CargarTFIDF()
  print 'Tabla TFIDF Cargada'
  cont=0
  for llave1 in TFIDF:
    for llave2 in TFIDF:
      if llave1 != llave2:
        SimCoseno = sim_coseno(TFIDF,llave1,llave2)
        print cont,llave1,llave2,SimCoseno[0]
        t=(cont,llave1,llave2,SimCoseno[0])
        c.execute('insert into SimTFIDF values (?,?,?,?)', t)
        #guardar en algo
      else:
        SimCoseno = 1
        print cont,llave1,llave2,SimCoseno
        t=(cont,llave1,llave2,SimCoseno)
        c.execute('insert into SimTFIDF values (?,?,?,?)', t)
      #Guardar en Base De Datos llave1, llave2 y SimCoseno
      cont=cont+1
  conn.commit()
  # We can also close the cursor if we are done with it
  c.close()
  return 0
##############################################################################
# funcion de la similitud de coseno
def sim_coseno(prefs,p1,p2):
  # Get the list of mutually rated items
  if p1 not in prefs or p2 not in prefs:  return 0 , 0
  si={}
  for item in prefs[p1]:
    if item in prefs[p2]: si[item]=1

  # if they are no ratings in common, return 0
  items_in_common = len(si)
  if items_in_common == 0 : return 0 , 0

  # Sum calculations
  n=items_in_common
  #p1_avg = user_average(prefs,p1)
  #p2_avg = user_average(prefs,p2)

  #num=sum( [ (prefs[p1][it]-p1_avg) *(prefs[p2][it]-p2_avg) for it in si])
  #sum1=sum( [ pow( prefs[p1][it]-p1_avg ,2) for it in si])
  #sum2=sum( [ pow( prefs[p2][it]-p2_avg ,2) for it in si])
  num=sum( [ (prefs[p1][it]) *(prefs[p2][it]) for it in si])
  sum1=sum( [ pow( prefs[p1][it],2) for it in si])
  sum2=sum( [ pow( prefs[p2][it],2) for it in si])
  # Sums of all the preferences
  
  # Calculate r (Coseno score)
  den=sqrt(sum1)*sqrt(sum2)
  if den==0: return 0 , 0
  r=num/den
  return r , items_in_common

################################
## dic Es Un Diccionario De Diccionarios
## Las Llaves Del Diccionario Principal Son De 1 Hasta 1682
## Las Llaves De Los Diccionarios Internos Van Desde 1 Hasta 14167
## 1682 Peliculas
## 14167 Terminos
## nmov : numero de movie
## npal : 
# carga a diccionario la tabla tfidf que se creo con la herramienta
def CargarTFIDF(ArchivoTFIDF = '.\VectorMovielens'):
    dic={}
    nmov=0
    x=[0]
    for line in open(ArchivoTFIDF):
        if nmov not in x:
            l=line.split(';')
            dic.setdefault(nmov,{})
            for npal in range(len(l)):
                dic[nmov][npal+1]=float(l[npal])
        nmov=nmov+1
    return dic
##########################################



def topMatches(prefs,person,n=20,similarity=sim_pearson):
  scores=[(similarity(prefs,person,other),other)
                  for other in prefs if other!=person]
  # Sort the list so the highest scores appear at the top
  scores.sort()
  scores.reverse()
  return scores[0:n]

#pendiente
def getRecommendations(prefs,person,similarity=sim_pearson):
  totals={}
  simSums={}
  for other in prefs:
    # don't compare me to myself
    if other==person: continue
    sim=similarity(prefs,person,other)
    # ignore scores of zero or lower
    if sim <=0 : continue
    for item in prefs[other]:
    # only score movies I haven't seen yet
      if item not in prefs[person] or prefs[person][item]==0:
        # Similarity * Score
        totals.setdefault(item,0)
        totals[item]+=prefs[other][item]*sim
        # Sum of similarities
        simSums.setdefault(item,0)
        simSums[item]+=sim

  # Create the normalized list
  rankings=[(total/simSums[item],item) for item,total in totals.items()]

  # Return the sorted list
  rankings.sort()
  rankings.reverse()
  return rankings

def loadMovieLensBase(path='.' , fold = 5 ):
    # Get movie titles
    # Get movie titles
    movies={}
    for line in open(path+'\u.item'):
        (id,title)=line.split('|')[0:2]
        movies[int(id)]=title
    # Load data
    DataSets = []

    for i in range(1,fold+1):
        prefs={}
        for line in open(path+'\u'+str(i)+'.base' ):
            (user,movieid,rating,ts)=line.split('\t')
            prefs.setdefault(int(user),{})
            prefs[int(user)][int(movieid)]=float(rating)
        DataSets.append(prefs)
    return DataSets

def loadMovieLensTest(path='.' , fold = 5 ):
    DataSets = []
    for i in range(1,fold+1):
        test_file = open(path+'\u'+str(i)+'.test')
        test = [map(int,  lineline.split('\t')) for lineline in test_file ]
        DataSets.append(test)
    return DataSets


def dictToDense( prefs ,users = 943 , items = 1682 ):
    ratings = numpy.zeros((users+1,items+1))
    for user in prefs:
        for item in prefs[user]:
            ratings[user][item] = prefs[user][item]
    return  ratings


def transformPrefs(prefs):
  result={}
  for person in prefs:
    for item in prefs[person]:
      result.setdefault(item,{})

      # Flip item and person
      result[item][person]=prefs[person][item]
  return result


# Metrica RM1
def number_of_rated_items_by_user(prefs, user_id):
    if user_id not  in prefs:
        return 0
    else:
        return len(prefs[user_id])
# Metrica RM2
def number_of_users_that_rated_item(prefs, item_id):
    count = 0
    for person in prefs:
        if item_id in prefs[person]:
            count+=1
    return count

def user_average(prefs, user_id):
    if user_id not  in prefs:
        return None
    else:
        p = numpy.array( [prefs[user_id][item] for  item  in  prefs[user_id]])
        return p.mean()

def user_std(prefs, user_id):
    if user_id not  in prefs:
        return None
    else:
        p = numpy.array( [prefs[user_id][item] for  item  in  prefs[user_id]])
        return p.std()


def top_n_deviation(prefs, user_id, item_id ):
    if user_id not  in prefs: return None
    SumAverage = 0.0
    otherN = 0
    #Para todos los usuarios
    #print user_id,item_id
    for other in prefs:
    #Excepto el mismo
        if other==user_id: continue
        #Si la persona califico el item
        if item_id in prefs[other]:
            SumAverage+=prefs[other][item_id]-user_average(prefs,other)
            otherN+=1
    if not otherN:
        return None
    else:
        return user_average(prefs,user_id)+SumAverage/otherN

# MOTOR Filtrado Colaborativo
# de que se sacan los vecinos
# Metrica RM5.- Numero de vecinos de u que han votado a i in common
# revisar
def collaborative_filtering(prefs, user_id, item_id, similarity = sim_pearson, min_items=50):
    total = 0.0
    simSum = 0.0
    inCommon = []
    #Para todos los usuarios
    for other in prefs:
    #Excepto el mismo
        if other==user_id: continue
        #Si la persona califico el item
        if item_id in prefs[other]:
            sim, items_in_common =similarity(prefs,user_id,other)
            inCommon.append(items_in_common)
            if sim < 0 : continue
            if similarity.__name__ == 'sim_pearson':
                if items_in_common < min_items:
                    sim = sim/min_items
            # Similarity * Score
            total+=prefs[other][item_id]*sim
            # Sum of similarities
            simSum+=sim

    if simSum > 0  :
        return total/simSum , sum(inCommon)/len(inCommon)
    else:
    #TO DO:
    # Si no hay usuarios en comun
        return None, 0


def collaborative_filtering_std(prefs, user_id, item_id, similarity = sim_pearson, min_items=50):
    total = 0.0
    simSum = 0.0
    inCommon = []
    #Para todos los usuarios
    for other in prefs:
    #Excepto el mismo
        if other==user_id: continue
        #Si la persona califico el item
        if item_id in prefs[other]:
            sim, items_in_common =similarity(prefs,user_id,other)
            inCommon.append(items_in_common)
            if sim < 0 : continue
            if similarity.__name__ == 'sim_pearson':
                if items_in_common < min_items:
                    sim = sim/min_items
            # Similarity * Score

            ### IF std is Zero IGNORE ?????
            u_std = user_std(prefs,other)
            if u_std:
                total+= ((prefs[other][item_id]-user_average(prefs,other))/u_std)*sim
                # Sum of similarities
                simSum+=sim
            else: continue

    if simSum > 0  :
        return user_average(prefs,user_id)+ user_std(prefs,user_id)*total/simSum , sum(inCommon)/len(inCommon),simSum/len(inCommon)
    else:
    #TO DO:
    # Si no hay usuarios en comun
        return None, 0, 0

def genre_similarity(movie1, movie2):
    m1 = numpy.array( get_movie_genre(movie1, "Movielens.sqlite3") )
    m2 = numpy.array( get_movie_genre(movie2, "Movielens.sqlite3") )
    return   m1.dot(m2) / numpy.linalg.norm(m1)*numpy.linalg.norm(m2)

def item_based_genre( prefs, user_id, item_id , similarity=sim_pearson):
    total=0.0
    simSum=0.0
    #print user_id, item_id
    items = transformPrefs(prefs)
    for other_item in items:
    #Excepto el mismo
        if other_item==item_id: continue
        #Si la persona califico el item
        if user_id in items[other_item]:
            sim,users_in_common  =similarity(items,item_id,other_item)
            if sim <0 : continue
            # Similarity * Score
            total+=items[other_item][user_id]*sim
            # Sum of similarities
            simSum+=sim
    if simSum > 0  :
        return  total/simSum
    else:
        return None


#basado en item######
def item_based( prefs, user_id, item_id, n= 150,  similarity=sim_pearson, threshold = 0):
    if user_id not in prefs:
        return None ,0,0,0
    userRatings=prefs[user_id]
    if len(userRatings) <= 1:
        return None,0,0,0
    ## Is better to receive items , so its not calculated for each call
    items = transformPrefs(prefs)

    scores = [ (similarity(items,item_id,other),other) for other in userRatings ]
    scores.sort(reverse=True)
    k_scores = scores[:n]
    
    total_weights = sum([sim for ((sim,num),item) in k_scores if sim > threshold])

    if total_weights <= 0:
        return None,0,0,0

    NumberItemsWithSimilarityVeryHigh = len([sim for ((sim,num),item_id) in k_scores if sim >= 0.9])
    NumberItemsWithSimilarityHigh = len([sim for ((sim,num),item_id) in k_scores if sim >= 0.7])
    NumberItemsWithSimilarityMed = len([sim for ((sim,num),item_id) in k_scores if sim >= 0.5])


    rating = sum([sim*prefs[user_id][item_id] for ((sim,num),item_id) in k_scores if sim > threshold])/total_weights

    return rating, NumberItemsWithSimilarityVeryHigh, NumberItemsWithSimilarityHigh,NumberItemsWithSimilarityMed

#----falta hacer una igual para el basado en contenido----#    
def user_based( prefs, similarity, user_id, item_id ):
    total=0.0
    simSum=0.0
    #Para todos los usuarios
    #print user_id,item_id
    for other in prefs:
    #Excepto el mismo
        if other==user_id: continue
        #Si la persona califico el item
        if item_id in prefs[other]:
            sim=similarity(prefs,user_id,other)
            if sim <0 : continue
            # Similarity * Score
            total+=prefs[other][item_id]*sim
            # Sum of similarities
            simSum+=sim
    if simSum > 0  :
        return total/simSum
    else:
        return None


def get_movie_genre(movie_id, sqliteFile='Movielens.sqlite3'):
    conn = sqlite3.connect(sqliteFile)
    c = conn.cursor()
    conn.text_factory = str
    t = (movie_id,)
    c.execute(
        """select *
           from Movie
           where movie_id = ?""", t
        )
    genres = []
    for r in c:
        for i in range(5,23,1):
            genres.append(r[i])

    # Save (commit) the changes
    conn.commit()

    # We can also close the cursor if we are done with it
    c.close()
    return genres


def loadMovieLensBaseOrdered(sqliteFile , limit , offset = 0  ):
    prefs={}
    conn = sqlite3.connect(sqliteFile)
    c = conn.cursor()
    conn.text_factory = str
    t = (limit, offset)
    c.execute(
    """select *
       from ratings
       order by time_stamp
       limit ?
       offset ?""", t
    )
    for row in c:
        user_id = row[0]
        movie_id = row[1]
        rating = row[2]
        ts = row [3]
        prefs.setdefault(int(user_id),{})
        prefs[int(user_id)][int(movie_id)]=float(rating)
    return prefs

def loadMovieLensTestOrdered(sqliteFile , limit , offset = 0  ):
    conn = sqlite3.connect(sqliteFile)
    c = conn.cursor()
    conn.text_factory = str
    t = (limit, offset)
    c.execute(
    """select *
       from ratings
       order by time_stamp
       limit ?
       offset ?""", t
    )
    test = [  row    for row in c]
    return test


def loadMovieLensAll(path='.'):
    # Get movie titles
    movies={}
    for line in open(path+'\u.item'):
        (id,title)=line.split('|')[0:2]
        movies[int(id)]=title
    # Load data
    prefs={}
    for line in open(path+'\u.data'):
        (user,movieid,rating,ts)=line.split('\t')
        prefs.setdefault(int(user),{})
        prefs[int(user)][int(movieid)]=float(rating)
    return  movies, prefs


def foldValidation(DSBase,DSTest, recommender ,similarity = sim_distance):
    for f in range(len(DSBase)):
        prefs = DSBase[f]
        results = [ recommender(prefs,similarity,r[0],r[1],r[2]) for r in DSTest[f]]
        resultsArray = numpy.array(results)
        print sum(abs(resultsArray[:,2]-resultsArray[:,3]))/resultsArray.shape[0]


def test(DSBase,DSTest):
    prefs = DSBase[0]
    results = []
    return results

def testOrdered(db_file = None , append_to_prefs = True , iteration = 0 , limit = 1000, offset= 0 ):
    prefs = loadMovieLensBaseOrdered(db_file,  limit = (iteration+1)*limit, offset = 0)
    test  = loadMovieLensTestOrdered(db_file,  limit = limit, offset = (iteration+1)*limit)
    print limit, (iteration+1)*limit
    results = []
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    conn.text_factory = str

    for (user_id, item_id, real, ts) in test:
        nriu = number_of_rated_items_by_user(prefs,user_id)
        ua = user_average(prefs,user_id )
        tn = top_n_deviation(prefs,user_id , item_id)
        nuri = number_of_users_that_rated_item(prefs, item_id)
        cf , avg_items_in_common  = collaborative_filtering(prefs,user_id, item_id)
        cfstd, x, sim_avg  = collaborative_filtering_std(prefs,user_id, item_id)

        (ib, nivh,nih,nim) = item_based(prefs,user_id, item_id)

        preds = [tn,cfstd,ib]

        #TODO FIS
        #print nriu,sim_avg, nivh
        fis = FIS( ( nriu,sim_avg,nivh), {'User Average':ua, 'Top N':tn, 'CF':cf, 'Item Based':ib } )

#       fis = FIS( ( nriu,sim_avg, nivh), {'User Average':ua, 'Top N':tn, 'CF':cf, 'Item Based':ib } )
        #print preds
        # if at least one pred is not None
        if preds.count(None) < len(preds):
            best =preds[min([(abs(p-real),i) for i,p in enumerate(preds) if p is not None])[1]]
            best_index = preds.index(best)
        else:
            best = None
            best_index = None
        t = (iteration, user_id, item_id, real, ts, nriu, nuri,avg_items_in_common,ua,tn,cf,cfstd,sim_avg ,ib,  nivh,nih,nim, best , best_index, fis)

        c.execute(
          """
            INSERT INTO test_run
                ( iteration, user_id, movie_id, rating, ts, number_of_rated_items_by_user,
                number_of_users_that_rated_item, avg_items_in_common , user_average,
                top_n_deviation, collaborative_filtering,collaborative_filtering_std,sim_avg ,item_based, NumberItemsWithSimilarityVeryHigh, NumberItemsWithSimilarityHigh,NumberItemsWithSimilarityMed, best, best_index, fis )
                values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", t )

        results.append(t)
        if append_to_prefs:
            prefs.setdefault(int(user_id),{})
            prefs[int(user_id)][int(item_id)]=float(real)

    conn.commit()

    c.close()
    
    return results , prefs



def testRun(db_file = None, iterations = 1, size= 1000):
    #Preparamos BD
    if not db_file:
        db_file = datetime.datetime.today().isoformat(sep='_')[5:-7].replace(":","_")+".db3"
    db_setup(db_file)
    # Iteracion
    for i in range(iterations):
        testOrdered(db_file, iteration = i,limit = size)

    getTestRun(db_file)



def plotItemsByUser(db_file, max_iteration = 100):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    conn.text_factory = str
    c.execute(
          """
            select
            number_of_rated_items_by_user, count(user_average),count(top_n_deviation) ,count(collaborative_filtering_std), count(item_based), count(fis),
                sum(abs(rating-user_average))/count(user_average) user_average,
                sum(abs(rating-top_n_deviation))/count(top_n_deviation) top_n_deviation,
                sum(abs(rating-collaborative_filtering))/count(collaborative_filtering) collaborative_filtering,
                sum(abs(rating-collaborative_filtering_std))/count(collaborative_filtering) collaborative_filtering_std,
                sum(abs(rating-item_based))/count(item_based) item_based,
                sum(abs(rating-best))/count(best) best,
                sum(abs(rating-fis))/count(fis) fis,
                count(best),
                count(*)
                from test_run t
                where number_of_rated_items_by_user < 20
                group by number_of_rated_items_by_user
                order by number_of_rated_items_by_user""" )

    by_iteration = numpy.array( [row for row in c ] )

    fig = plt.figure(1)
    plt.subplot(211)
    l1,l2,l3,l4,l5,l6 = plt.plot( by_iteration[:,0],
              by_iteration[:,6],'b-',  # user average
              by_iteration[:,7],'c-' , # Top N
              by_iteration[:,9],'m-', # CF
              by_iteration[:,10],'y-',   # Item Based
              by_iteration[:,11],'r-', #BEST
              by_iteration[:,12],'k-', #FIS
              lw=1 )
    plt.ylabel('MAE')
    plt.xlabel('Number of Items Rated by User')
    plt.grid(True)
    fig.legend((l1,l2,l3,l4,l5,l6) , ('User Average', 'Top N', 'CF', 'Item Based','Best','FIS'), loc='upper right')

    c.execute(
            """
            select
            number_of_rated_items_by_user,
            count(*)
            from test_run t
            group by number_of_rated_items_by_user
            order by number_of_rated_items_by_user""" )

    num = numpy.array( [row for row in c ] )

    plt.subplot(212)
    plt.plot( num[:,0],
              num[:,1],'b-',  # user average
              lw=1 )
    plt.ylabel('Number of Users')
    plt.xlabel('Number of Items Rated by User')
    plt.grid(True)

    plt.show()
    return by_iteration

def plotSimilarItemsByBestPredictor(db_file, rated_items = 20, sim_avg = 0.09):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    conn.text_factory = str
    t= (rated_items, sim_avg)
    c.execute("""select NumberItemsWithSimilarityVeryHigh, best_index, count(*)
                 from test_run t
                 where number_of_rated_items_by_user < ? and sim_avg <?
                 and NumberItemsWithSimilarityVeryHigh >= 0
                 group by NumberItemsWithSimilarityVeryHigh , best_index""" ,t
    )

    outerJoin = {}
    max_value = 0
    for row in c:
        outerJoin.setdefault(row[0],{})
        outerJoin[row[0]][row[1]]=row[2]
        if max_value < row[0]:
            max_value = row[0]

    data = []
    for i in range(int(max_value)+1):
        if i in outerJoin:
            preds = [i]
            for p in range(4):
                if p in outerJoin[i]:
                    preds.append(outerJoin[i][p])
                else:
                    preds.append(0)
            data.append(preds)

        else:
            data.append([i,0,0,0,0])

    num_similar_items = numpy.array(data )
    fig = plt.figure(1)

    #l1,l2,
    l3,l4 = plt.plot( num_similar_items[:,0],
              #num_similar_items[:,1],'b-',  # user average
              #num_similar_items[:,2],'c-' , # Top N
              num_similar_items[:,3],'m-', # CF
              num_similar_items[:,4],'y-',   # Item Based
              lw=1 )
    plt.ylabel('Number of best predictions')
    plt.xlabel('number of highly similar items, s < .70')
    plt.grid(True)

#    fig.legend((l1,l2,l3,l4) , ('User Average', 'Top N', 'CF', 'Item Based'), loc='upper right')
    fig.legend((l3,l4) , ( 'CF', 'Item Based'), loc='upper right')

    plt.show()
    
    return num_similar_items




def printTestRun(db_file, max_iteration = 100):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    conn.text_factory = str
    c.execute(
          """select iteration, count(user_average),count(top_n_deviation) ,count(collaborative_filtering_std), count(item_based), count(fis),
                sum(abs(rating-user_average))/count(user_average) user_average,
                sum(abs(rating-top_n_deviation))/count(top_n_deviation) top_n_deviation,
                sum(abs(rating-collaborative_filtering))/count(collaborative_filtering) collaborative_filtering,
                sum(abs(rating-collaborative_filtering_std))/count(collaborative_filtering) collaborative_filtering_std,
                sum(abs(rating-item_based))/count(item_based) item_based,
                sum(abs(rating-best))/count(best) best,
                sum(abs(rating-fis))/count(fis) fis,
                count(best),
                count(*)
                from test_run t
                group by iteration""" )

    by_iteration = numpy.array( [row for row in c ] )

    fig = plt.figure(1)
    plt.subplot(211)
    l1,l2,l3,l4,l5,l6 = plt.plot( by_iteration[:,0],
              by_iteration[:,6],'r-',  # user average
              by_iteration[:,7],'c-' , # Top N
              by_iteration[:,9],'m-', # CF
              by_iteration[:,10],'y-',   # Item Based
              by_iteration[:,11],'k-', #BEST
              by_iteration[:,12],'b-', #FIS
              lw=2 )
    plt.ylabel('MAE')
    plt.xlabel('Iteration')
    plt.grid(True)
    fig.legend((l1,l2,l3,l4,l5,l6) , ('User Average', 'Top N', 'CF', 'Item Based','Best','FIS'), loc='upper right')

    plt.subplot(212)
    plt.plot( by_iteration[:,0],
              by_iteration[:,1],'r-',  # user average
              by_iteration[:,2],'c-' , # Top N
              by_iteration[:,3],'m-', # CF
              by_iteration[:,4],'y-',   # Item Based
              by_iteration[:,5],'b-',   # FIS
              lw=2 )
    plt.ylabel('Coverage')
    plt.xlabel('Iteration')
    plt.grid(True)

#    c.execute(
#             """select
#                user_id,
#                count(user_average),count(top_n_deviation) ,count(collaborative_filtering), count(item_based),
#                sum(abs(rating-user_average))/count(user_average) user_average,
#                sum(abs(rating-top_n_deviation))/count(top_n_deviation)  top_n_deviation,
#
#                sum(abs(rating-collaborative_filtering))/count(*) collaborative_filtering,
#                sum(abs(rating-collaborative_filtering_std))/count(collaborative_filtering) collaborative_filtering_std,
#                sum(abs(rating-item_based))/count(item_based) item_based,
#
#
#                sum(abs(rating-best))/count(best) best ,
#
#                count(best),
#                count(*)
#                from test_run t
#                ---group by iteration
#                group by user_id
#                order by user_id""" )
#
#    by_iteration = numpy.array( [row for row in c ] )
#
#    fig = plt.figure(2)
#    plt.subplot(211)
#    l1,l2,l3,l4,l5 = plt.plot( by_iteration[:,0],
#              by_iteration[:,5],'b-',  # user average
#              by_iteration[:,6],'c-' , # Top N
#              by_iteration[:,8],'m-', # CF
#              by_iteration[:,9],'y-',   # Item Based
#              by_iteration[:,10],'r-', #BEST
#              lw=1 )
#    plt.ylabel('MAE')
#    plt.xlabel('Number of rated items')
#    plt.grid(True)
#    fig.legend((l1,l2,l3,l4,l5) , ('User Average', 'Top N', 'CF', 'Item Based','Best'), loc='upper right')

    plt.show()

    t = (max_iteration,)
    c.execute(
          """select *
             from test_run t
             where iteration <= ? """,t )
    test_run = numpy.array( [row for row in c ] )


    return by_iteration, test_run



def printTestRun2( max_iteration = 100):

    totalMAE, by_iteration , recoms, rats, W1,W2 = FIS_Test3(instances=99000, return_datasets=True)

    fig = plt.figure(1)
    plt.subplot(211)
    l1,l2,l3,l4,l5,l6 = plt.plot( by_iteration[:,0],
              by_iteration[:,6],'b-',  # user average
              by_iteration[:,7],'c-' , # Top N
              by_iteration[:,9],'m-', # CF
              by_iteration[:,10],'y-',   # Item Based
              by_iteration[:,11],'r-', #BEST
              by_iteration[:,20],'k-', #FIS
              lw=1 )
    plt.ylabel('MAE')
    plt.xlabel('Iteration')
    plt.grid(True)
    fig.legend((l1,l2,l3,l4,l5,l6) , ('User Average', 'Top N', 'CF', 'Item Based','Best','FIS'), loc='upper right')

    plt.subplot(212)
    plt.plot( by_iteration[:,0],
              by_iteration[:,1],'b-',  # user average
              by_iteration[:,2],'c-' , # Top N
              by_iteration[:,3],'m-', # CF
              by_iteration[:,4],'y-',   # Item Based
              by_iteration[:,5],'k-',   # FIS
              lw=1 )
    plt.ylabel('Coverage')
    plt.xlabel('Iteration')
    plt.grid(True)

#    c.execute(
#             """select
#                user_id,
#                count(user_average),count(top_n_deviation) ,count(collaborative_filtering), count(item_based),
#                sum(abs(rating-user_average))/count(user_average) user_average,
#                sum(abs(rating-top_n_deviation))/count(top_n_deviation)  top_n_deviation,
#
#                sum(abs(rating-collaborative_filtering))/count(*) collaborative_filtering,
#                sum(abs(rating-collaborative_filtering_std))/count(collaborative_filtering) collaborative_filtering_std,
#                sum(abs(rating-item_based))/count(item_based) item_based,
#
#
#                sum(abs(rating-best))/count(best) best ,
#
#                count(best),
#                count(*)
#                from test_run t
#                ---group by iteration
#                group by user_id
#                order by user_id""" )
#
#    by_iteration = numpy.array( [row for row in c ] )
#
#    fig = plt.figure(2)
#    plt.subplot(211)
#    l1,l2,l3,l4,l5 = plt.plot( by_iteration[:,0],
#              by_iteration[:,5],'b-',  # user average
#              by_iteration[:,6],'c-' , # Top N
#              by_iteration[:,8],'m-', # CF
#              by_iteration[:,9],'y-',   # Item Based
#              by_iteration[:,10],'r-', #BEST
#              lw=1 )
#    plt.ylabel('MAE')
#    plt.xlabel('Number of rated items')
#    plt.grid(True)
#    fig.legend((l1,l2,l3,l4,l5) , ('User Average', 'Top N', 'CF', 'Item Based','Best'), loc='upper right')

    plt.show()

    t = (max_iteration,)
    c.execute(
          """select *
             from test_run t
             where iteration <= ? """,t )
    test_run = numpy.array( [row for row in c ] )


    return by_iteration, test_run



def getTestRun(db_file, max_iteration = 100):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    conn.text_factory = str
    t = (max_iteration,)
    c.execute(
          """select *
             from test_run t
             where iteration <= ? """,t )
    test_run = numpy.array( [row for row in c ] )
    return test_run




def FIS_Test3(instances = 1000, return_datasets= False):
    test = getTestRun("1kTestRun3.db3")
    #test = getTestRun("1kTestRun3LessThan30.db3")
    W1 = map(RecommenderEval.EvalFisSingle, zip(test[:instances,5],test[:instances,8]))
    W2 = map( RecommenderItemCF2.EvalFisSingle, zip(test[:instances,5],test[:instances,8],test[:instances,14]))
    Weights = [t[0]+t[1] for t in  zip(W1,W2)]
    #W1 (TopNw, CFw)
    #W2 (ItemBasedw , CFw)
    Recs = map(Weighted_avg2,Weights, zip(test[:instances,12],test[:instances,10],test[:instances,11],test[:instances,10],test[:instances,9]))
    k = test[:instances,:]
    k = numpy.column_stack([k, Recs])
    import numpy.ma as ma
    recomendaciones =  ma.masked_equal(k[:instances,20],None)
    ratings = ma.array(k[:,3], mask = recomendaciones.mask)
    r = sum( numpy.abs( recomendaciones.compressed()-ratings.compressed() ) ) / len(ratings.compressed())
    if return_datasets:
        return r, k, recomendaciones, ratings, W1, W2
    else:
        return r

def PlotCompByNumberOfRatings( numbers ):

    totalMAE, k, recoms, rats, W1,W2 = FIS_Test3(instances=99000, return_datasets=True)
    TopN= k[:,10]
    CF  = k[:,12]
    IB  = k[:,13]
    FIS = k[:,20]

    TopN_MAE = []
    CF_MAE = []
    IB_MAE = []
    FIS_MAE = []

    TopN_Coverage = []
    CF_Coverage = []
    IB_Coverage = []
    FIS_Coverage = []

    user_ratings = k[:,3]

    for number in numbers:
        TotalLessThan =  numpy.equal(k[:,5],number)
        TotalNumber = float(len(k[TotalLessThan]))
    #CF
        #Only Valid Recommendations, (Not None, in this case None is equal to Zero)
        LessThanCF  = numpy.logical_and(CF > 0, k[:,5] == number)
        CF_MAE.append( sum( numpy.abs( CF[LessThanCF]-user_ratings[LessThanCF] ) ) / len(CF[LessThanCF]) )
        CF_Coverage.append(100*len(CF[LessThanCF])/TotalNumber)
    #IB
        #Only Valid Recommendations, (Not None, in this case None is equal to Zero)
        LessThanIB  = numpy.logical_and( IB > 0, k[:,5] == number)
        IB_MAE.append( sum( numpy.abs( IB[LessThanIB]-user_ratings[LessThanIB] ) ) / len(IB[LessThanIB]) )
        IB_Coverage.append(100*len(IB[LessThanIB])/TotalNumber)

    #TopN
        #Only Valid Recommendations, (Not None, in this case None is equal to Zero)
        LessThanTopN  = numpy.logical_and( TopN > 0, k[:,5] == number)
        TopN_MAE.append( sum( numpy.abs( TopN[LessThanTopN]-user_ratings[LessThanTopN] ) ) / len(TopN[LessThanIB]) )
        TopN_Coverage.append(100*len(TopN[LessThanTopN])/TotalNumber)

    #FIS
        #Only Valid Recommendations, (Not None, in this case None is equal to Zero)
        LessThanFIS  = numpy.logical_and( FIS > 0, k[:,5] == number)
        FIS_MAE.append( sum( numpy.abs( FIS[LessThanFIS]-user_ratings[LessThanFIS] ) ) / len(FIS[LessThanFIS]) )
        FIS_Coverage.append(100*len(FIS[LessThanFIS])/TotalNumber)



    fig = plt.figure(1)
    plt.subplot(211)
    l1,l2,l3,l4 = plt.plot( numbers, CF_MAE,'b->',
                            numbers, IB_MAE,'c-o',
                            numbers, TopN_MAE,'m-<',
                            numbers, FIS_MAE,'y-',
                         lw=1 )
    plt.ylabel('MAE')
    plt.xlabel('Number of Items Rated by User')
    plt.grid(True)

    fig.legend((l1,l2,l3,l4) , ('CF', 'Item Based','Top N','FIS'), loc='upper right')

    plt.subplot(212)
    plt.plot( numbers, CF_Coverage,'b->',
              numbers, IB_Coverage,'c-o',
              numbers, TopN_Coverage,'m-<',
              numbers, FIS_Coverage,'y-',
                                       lw=1 )
    plt.ylabel('Coverage')
    plt.xlabel('Number of Items Rated by User')
    plt.grid(True)

    plt.show()

    return CF_MAE,IB_MAE,TopN, FIS_MAE,W1,W2



def PlotCompByNumberOfRatingsB( numbers ):
    totalMAE, k, recoms, rats, W1,W2 = FIS_Test3(instances=99000, return_datasets=True)
    TopN= k[:,10]
    CF  = k[:,12]
    IB  = k[:,13]
    FIS = k[:,20]

    TopN_MAE = []
    CF_MAE = []
    IB_MAE = []
    FIS_MAE = []

    TopN_Coverage = []
    CF_Coverage = []
    IB_Coverage = []
    FIS_Coverage = []

    user_ratings = k[:,3]

    for number in numbers:
        TotalLessThan =  numpy.less_equal(k[:,5],number)
        TotalNumber = float(len(k[TotalLessThan]))
    #CF
        #Only Valid Recommendations, (Not None, in this case None is equal to Zero)
        BothCF  = numpy.logical_and(CF > 0, FIS > 0, )
        LessThanCF  = numpy.logical_and( BothCF, k[:,5] < number)
        CF_MAE.append( sum( numpy.abs( CF[LessThanCF]-user_ratings[LessThanCF] ) ) / len(CF[LessThanCF]) )
        CF_Coverage.append(100*len(CF[LessThanCF])/TotalNumber)
    #IB
        #Only Valid Recommendations, (Not None, in this case None is equal to Zero)
        LessThanIB  = numpy.logical_and( IB > 0, k[:,5] < number)
        IB_MAE.append( sum( numpy.abs( IB[LessThanIB]-user_ratings[LessThanIB] ) ) / len(IB[LessThanIB]) )
        IB_Coverage.append(100*len(IB[LessThanIB])/TotalNumber)

    #TopN
        #Only Valid Recommendations, (Not None, in this case None is equal to Zero)
        LessThanTopN  = numpy.logical_and( TopN > 0, k[:,5] < number)
        TopN_MAE.append( sum( numpy.abs( TopN[LessThanTopN]-user_ratings[LessThanTopN] ) ) / len(TopN[LessThanIB]) )
        TopN_Coverage.append(100*len(TopN[LessThanTopN])/TotalNumber)

    #FIS
        #Only Valid Recommendations, (Not None, in this case None is equal to Zero)
        LessThanFIS  = numpy.logical_and( FIS > 0, k[:,5] < number)
        FIS_MAE.append( sum( numpy.abs( FIS[LessThanFIS]-user_ratings[LessThanFIS] ) ) / len(FIS[LessThanFIS]) )
        FIS_Coverage.append(100*len(FIS[LessThanFIS])/TotalNumber)



    fig = plt.figure(1)
    plt.subplot(211)
    l1,l2,l3,l4 = plt.plot( numbers, CF_MAE,'b->',
                            numbers, IB_MAE,'c-^',
                            numbers, TopN_MAE,'m-o',
                            numbers, FIS_MAE,'y-o',
                         lw=1 )
    plt.ylabel('MAE')
    plt.xlabel('Maximum Number of Items Rated by User')
    plt.grid(True)

    fig.legend((l1,l2,l3,l4) , ('CF', 'Item Based','Top N','FIS'), loc='upper right')

    plt.subplot(212)
    plt.plot( numbers, CF_Coverage,'b-',
              numbers, IB_Coverage,'c-',
              numbers, TopN_Coverage,'m-o',
              numbers, FIS_Coverage,'y-o',
                                       lw=1 )
    plt.ylabel('Coverage')
    plt.xlabel('Maximum Number of Items Rated by User')
    plt.grid(True)

    plt.show()

    return CF_MAE,IB_MAE,TopN, FIS_MAE,W1,W2





def FIS_Test(instances = 1000):
    test = getTestRun("1kTestRun3.db3")
    Weights = map(Recommender.eval, zip(test[:instances,5],test[:instances,8]))
    Recs = map(Weighted_avg,Weights,test[:instances,3],test[:instances,12],test[:instances,10])
    k = test[:instances,:]
    k = numpy.column_stack([k, Recs])
    import numpy.ma as ma
    recoms =  ma.masked_equal(k[:instances,20],None)
    rats = ma.array(k[:,3], mask = recoms.mask)
    r = sum( numpy.abs( recoms.compressed()-rats.compressed() ) ) / len(rats.compressed())
    return r ,k

def FIS_Test2(instances = 1000):
    test = getTestRun("1kTestRun3.db3")
    Weights = map(RecommenderItemCF.eval, zip(test[:instances,5],test[:instances,8],test[:instances,14]))
    Recs = map(Weighted_avg,Weights,test[:instances,3],test[:instances,12],test[:instances,10])
    k = test[:instances,:]
    k = numpy.column_stack([k, Recs])
    import numpy.ma as ma
    recoms =  ma.masked_equal(k[:instances,20],None)
    rats = ma.array(k[:,3], mask = recoms.mask)
    r = sum( numpy.abs( recoms.compressed()-rats.compressed() ) ) / len(rats.compressed())
    return r ,k

def FIS2(input_variables,predictors):
    #if input_variables[0] <10 and input_variables[1]< .003:
    #    return predictors['Item Based'] or predictors['User Average'] or None
    #( nriu,sim_avg,nivh)

    W1 = RecommenderEval3.EvalFisSingle(input_variables[0],input_variables[1])
    W2 = RecommenderItemCF2.EvalFisSingle(input_variables[0],input_variables[1],input_variables[2])
    Weights = [t[0]+t[1] for t in  zip(W1,W2)]
    predictions =(predictors['Top N'],predictors['CF'],predictors['IB'],predictors['CF'],predictors['User Average'])
    recommendation = Weighted_avg2(Weights,predictions)
    return recommendation

def FIS(input_variables,predictors):
    if input_variables[0] <10 and input_variables[1]< .003:
        return predictors['Item Based'] or predictors['User Average'] or None
    if predictors['CF'] and predictors['Top N']:
        W = Recommender.eval(input_variables[:2] )
        P = [predictors['Top N'],predictors['CF']]
        rating =  sum([i*j for (i,j) in zip(W,P)]) /sum(W)
        return rating
    else:
        return  predictors['Top N'] or predictors['CF'] or predictors['User Average'] or None


def Weighted_avg( w, real, cf, top_n):
    if cf and top_n and sum(w):
        return (w[0]*cf + w[1]*top_n)/sum(w)
    else:
        return  top_n or cf or None

def Weighted_avg2( weights,predictions,simple = True):
    """
    weights[0]: TopNw (FIS1)
    weights[1]: CFw (FIS1)
    weights[2]: IBw (FIS2)
    weights[3]: CFw (FIS2)

    predictions[0]: UserAverage

    predictions[1]: TopN
    predictions[2]: CF
    predictions[3]: IB
    predictions[4]: CF

    """
    # All predictions are None?
    if len([1 for p in predictions if p]) == 0:
        return None
    # Weights SemiFinals



    user_avg = predictions[4]
    weights = list(weights[:4])
    predictions = list(predictions[:4])

    for i,v in enumerate(predictions):
        if v is None:
            predictions[i]=0
            weights[i]=0

#    best = max(weights)
#    best_index = weights.index(best)

#    return predictions[best_index]

    if sum(weights):
        result = sum([i*j for (i,j) in zip(weights,predictions)])/sum(weights)
    else:
        result =  None
        #result = user_avg  or None
    return result

def WinnerTakesAll( weights,predictions,simple = True):
    """
    weights[0]: TopNw (FIS1)
    weights[1]: CFw (FIS1)
    weights[2]: IBw (FIS2)
    weights[3]: CFw (FIS2)

    predictions[0]: UserAverage

    predictions[1]: TopN
    predictions[2]: CF
    predictions[3]: IB
    predictions[4]: CF

    """
    # All predictions are None?
    if len([1 for p in predictions if p]) == 0:
        return None
    # Weights SemiFinals

    weights = list(weights)
    predictions = list(predictions)

    for i,v in enumerate(predictions):
        if v is None:
            predictions[i]=0
            weights[i]=0

    best = max(weights)
    best_index = weights.index(best)

    return predictions[best_index]


if __name__ == "__main__":
    r = plotSimilarItemsByBestPredictor("1kTestRun2.db3")
    print r
#s= [((0.9045340337332909, 4), 258), ((0.8703882797784892, 4), 286), ((0.7484551991837513, 5), 7), ((0.7071067811865475, 4),300),
# ((0.5477225575051642, 6), 50),
# ((0.17407765595569785, 6), 117), ((0.04682929057908325, 7), 1), ((0.0, 3), 275), ( (0, 0), 864), ((0, 0), 748), ((0, 0), 678), ((0, 0), 628), ((0, 0), 515), ((0, 0
#), 326), ((0, 0), 283), ((0, 0), 137), ((0, 0), 129), ((0, 0), 116), ((0, 0), 25), ((-0.09090909090909091, 4), 294), ((-0.27735009811261496, 6), 100), ((-0.30151134457776363, 4),24),((-0.48420012470625223, 4), 475), ((-0.5, 3), 245), ((-0.5000000000000013, 3), 284), ((-0.7559289460184531, 3), 919),
#                     ((-0.9707253433941485, 3), 124)]
