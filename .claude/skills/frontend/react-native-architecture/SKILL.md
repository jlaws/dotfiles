---
name: react-native-architecture
description: Build production React Native apps with Expo, navigation, native modules, offline sync, and cross-platform patterns. Use when developing mobile apps, implementing native integrations, or architecting React Native projects.
---

# React Native Architecture

## Project Structure

```
src/
├── app/                    # Expo Router screens
│   ├── (auth)/            # Auth group
│   ├── (tabs)/            # Tab navigation
│   └── _layout.tsx        # Root layout
├── components/
│   ├── ui/                # Reusable UI components
│   └── features/          # Feature-specific components
├── hooks/                 # Custom hooks
├── services/              # API and native services
├── stores/                # State management
└── types/                 # TypeScript types
```

## Pattern 1: Expo Router Navigation

```typescript
// app/(tabs)/_layout.tsx
export default function TabLayout() {
  const { colors } = useTheme()
  return (
    <Tabs screenOptions={{
      tabBarActiveTintColor: colors.primary,
      headerShown: false,
    }}>
      <Tabs.Screen name="index" options={{ title: 'Home', tabBarIcon: ({ color, size }) => <Home size={size} color={color} /> }} />
      <Tabs.Screen name="search" options={{ title: 'Search', tabBarIcon: ({ color, size }) => <Search size={size} color={color} /> }} />
    </Tabs>
  )
}

// Programmatic navigation
import { router } from 'expo-router'
router.push('/profile/123')
router.replace('/login')
router.push({ pathname: '/product/[id]', params: { id: '123' } })
```

## Pattern 2: Authentication Flow

```typescript
// providers/AuthProvider.tsx
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const segments = useSegments()
  const router = useRouter()

  useEffect(() => { checkAuth() }, [])

  // Protect routes
  useEffect(() => {
    if (isLoading) return
    const inAuthGroup = segments[0] === '(auth)'
    if (!user && !inAuthGroup) router.replace('/login')
    else if (user && inAuthGroup) router.replace('/(tabs)')
  }, [user, segments, isLoading])

  async function signIn(credentials: Credentials) {
    const { token, user } = await api.login(credentials)
    await SecureStore.setItemAsync('authToken', token)
    setUser(user)
  }

  async function signOut() {
    await SecureStore.deleteItemAsync('authToken')
    setUser(null)
  }
  // ...
}
```

## Pattern 3: Offline-First with React Query

```typescript
import { onlineManager } from '@tanstack/react-query'
import NetInfo from '@react-native-community/netinfo'

onlineManager.setEventListener((setOnline) =>
  NetInfo.addEventListener((state) => setOnline(!!state.isConnected))
)

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { gcTime: 1000 * 60 * 60 * 24, staleTime: 1000 * 60 * 5, networkMode: 'offlineFirst' },
    mutations: { networkMode: 'offlineFirst' },
  },
})

const asyncStoragePersister = createAsyncStoragePersister({
  storage: AsyncStorage, key: 'REACT_QUERY_OFFLINE_CACHE',
})

// Optimistic update pattern
export function useCreateProduct() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: api.createProduct,
    onMutate: async (newProduct) => {
      await queryClient.cancelQueries({ queryKey: ['products'] })
      const previous = queryClient.getQueryData(['products'])
      queryClient.setQueryData(['products'], (old: Product[]) => [...old, { ...newProduct, id: 'temp-' + Date.now() }])
      return { previous }
    },
    onError: (err, newProduct, context) => queryClient.setQueryData(['products'], context?.previous),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['products'] }),
  })
}
```

## Pattern 4: Native Module Integration

```typescript
// services/biometrics.ts
export async function authenticateWithBiometrics(): Promise<boolean> {
  const hasHardware = await LocalAuthentication.hasHardwareAsync()
  if (!hasHardware) return false
  const isEnrolled = await LocalAuthentication.isEnrolledAsync()
  if (!isEnrolled) return false
  const result = await LocalAuthentication.authenticateAsync({
    promptMessage: 'Authenticate to continue',
    fallbackLabel: 'Use passcode',
  })
  return result.success
}

// services/notifications.ts
export async function registerForPushNotifications() {
  if (Platform.OS === 'android') {
    await Notifications.setNotificationChannelAsync('default', {
      name: 'default', importance: Notifications.AndroidImportance.MAX,
    })
  }
  const { status } = await Notifications.getPermissionsAsync()
  if (status !== 'granted') {
    const { status: newStatus } = await Notifications.requestPermissionsAsync()
    if (newStatus !== 'granted') return null
  }
  const projectId = Constants.expoConfig?.extra?.eas?.projectId
  return (await Notifications.getExpoPushTokenAsync({ projectId })).data
}
```

## Pattern 5: Performance Optimization

```typescript
import { FlashList } from '@shopify/flash-list'

const ProductItem = memo(function ProductItem({ item, onPress }: { item: Product; onPress: (id: string) => void }) {
  const handlePress = useCallback(() => onPress(item.id), [item.id, onPress])
  return (
    <Pressable onPress={handlePress} style={styles.item}>
      <FastImage source={{ uri: item.image }} style={styles.image} resizeMode="cover" />
      <Text>{item.name}</Text>
    </Pressable>
  )
})

export function ProductList({ products, onProductPress }: ProductListProps) {
  const renderItem = useCallback(({ item }: { item: Product }) => <ProductItem item={item} onPress={onProductPress} />, [onProductPress])
  return (
    <FlashList
      data={products} renderItem={renderItem} estimatedItemSize={100}
      removeClippedSubviews={true} maxToRenderPerBatch={10} windowSize={5}
    />
  )
}
```

## EAS Build & Submit

```json
// eas.json
{
  "build": {
    "development": { "developmentClient": true, "distribution": "internal", "ios": { "simulator": true } },
    "preview": { "distribution": "internal", "android": { "buildType": "apk" } },
    "production": { "autoIncrement": true }
  }
}
```

```bash
eas build --platform ios --profile development
eas build --platform all --profile production
eas submit --platform ios
eas update --branch production --message "Bug fixes"
```

## Platform-Specific Styles

```typescript
const styles = StyleSheet.create({
  button: {
    ...Platform.select({
      ios: { shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 4 },
      android: { elevation: 4 },
    }),
  },
})
```
